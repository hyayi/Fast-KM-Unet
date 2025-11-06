# train_qat.py
import argparse, os, random, json, shutil, yaml
from collections import OrderedDict
import numpy as np, pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import albumentations as A

# --- [QAT 추가] ---
import nncf
from nncf.config import NNCFConfig
import openvino as ov
# ------------------

import archs, losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter

# ------------------------- Utils -------------------------
# (기존 train.py와 동일: list_type, seed_all, is_dist_env,
#  init_distributed, cleanup_distributed, ddp_allreduce_mean,
#  build_criterion, save_ckpt, load_ckpt)
# ... (이하 모든 유틸리티 함수는 train.py와 동일) ...
def list_type(s):
    return [int(a) for a in s.split(',')]

def seed_all(seed=1029):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

def is_dist_env():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def init_distributed(backend="nccl"):
    if not is_dist_env():
        return False, 0, 0, 1
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def ddp_allreduce_mean(tensors, device):
    if not dist.is_initialized():
        return tensors
    keys = sorted(tensors.keys())
    vec = torch.tensor([float(tensors[k]) for k in keys],
                       device=device, dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec /= dist.get_world_size()
    return {k: v.item() for k, v in zip(keys, vec)}

def build_criterion(name):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().cuda()
    return losses.__dict__[name]().cuda()

def save_ckpt(path, model, optimizer, scheduler, epoch, best_iou, best_dice, config):
    model_to_save = model.module if hasattr(model, "module") else model
    
    # NNCF QAT 모델 저장 시: controller.get_compression_state() 포함
    state_to_save = {
        'model': model_to_save.state_dict(),
    }
    
    # NNCF Controller가 있는지 확인 (QAT 모델인 경우)
    if hasattr(model, 'controller'):
        print("[NNCF] Saving compression state...")
        state_to_save['compression_state'] = model.controller.get_compression_state()

    ckpt = {
        'epoch': epoch,
        'state_dict': state_to_save, # [수정] 모델과 압축 상태를 함께 저장
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_iou': best_iou,
        'best_dice': best_dice,
        'config': config,
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, strict=True, 
              nncf_compression_state=None):
    
    ckpt = torch.load(path, map_location='cuda')
    
    # [수정] NNCF QAT 체크포인트 로드
    state_dict_container = ckpt.get('state_dict')
    if state_dict_container:
        state = state_dict_container['model']
        compression_state = state_dict_container.get('compression_state')
    else:
        # (하위 호환)
        state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
        compression_state = None
        
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        
    model.load_state_dict(state, strict=strict)
    
    if optimizer is not None and isinstance(ckpt.get('optimizer'), dict):
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and isinstance(ckpt.get('scheduler'), dict):
        scheduler.load_state_dict(ckpt['scheduler'])
        
    # [수정] NNCF 압축 상태 로드
    if nncf_compression_state is not None and compression_state is not None:
        print("[NNCF] Loading compression state from checkpoint...")
        nncf_compression_state.load_state_dict(compression_state)

    start_epoch = int(ckpt.get('epoch', 0)) + 1
    best_iou = float(ckpt.get('best_iou', 0.0))
    best_dice = float(ckpt.get('best_dice', 0.0))
    return start_epoch, best_iou, best_dice


# ------------------------- Data -------------------------
# (기존 train.py와 동일: make_dataloaders)
# ...
def make_dataloaders(cfg, distributed, img_ext='_0000.nii.gz', mask_ext='.png'):
    if cfg['dataset'] == 'busi':
        mask_ext = '_mask.png'
    elif cfg['dataset'] in ['glas','cvc','isic2018','isic2017']:
        mask_ext = '.png'
    elif cfg['dataset'] == 'ngtube':
        mask_ext = '.nii.gz'

    with open(cfg["splits_final"]) as f:
        sp = json.load(f)
        train_ids, val_ids = sp[0]["train"], sp[0]["val"]

    train_tf = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])
    val_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    train_ds = Dataset(train_ids, cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=train_tf)
    val_ds   = Dataset(val_ids,   cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=val_tf)

    dl_common = dict(batch_size=cfg['batch_size'], pin_memory=True)
    if cfg['num_workers'] > 0:
        dl_common.update(dict(
            num_workers=cfg['num_workers'],
            persistent_workers=True,
            prefetch_factor=8
        ))
    else:
        dl_common.update(dict(num_workers=0))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_ds, sampler=train_sampler, shuffle=False, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   sampler=val_sampler,   shuffle=False, drop_last=False, **dl_common
        )
    else:
        train_sampler = val_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_ds, shuffle=True, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   shuffle=False, drop_last=False, **dl_common
        )

    return train_loader, val_loader, train_sampler, val_sampler

# ------------------------- Optim/Sched -------------------------
# (기존 train.py와 동일: build_optimizer, build_scheduler)
# ...
def build_optimizer(cfg, model):
    kan_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('layer' in n.lower()) and ('fc' in n.lower()):
            kan_params.append(p)
        else:
            base_params.append(p)
    groups = [
        {'params': base_params, 'lr': cfg['lr'],      'weight_decay': cfg['weight_decay']},
        {'params': kan_params,  'lr': cfg['kan_lr'], 'weight_decay': cfg['kan_weight_decay']},
    ]
    if cfg['optimizer'] == 'Adam':
        return optim.Adam(groups)
    return optim.SGD(groups, lr=cfg['lr'], momentum=cfg['momentum'],
                     nesterov=cfg['nesterov'], weight_decay=cfg['weight_decay'])

def build_scheduler(cfg, optimizer):
    if cfg['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    # ... (기타 스케줄러)
    raise NotImplementedError


# ------------------------- Train/Valid (AMP 지원) -------------------------
# (기존 train.py와 동일: select_amp_dtype, train_one_epoch, validate_one_epoch)
# ...
def select_amp_dtype(mode: str):
    if mode == "off": return None
    if mode == "bf16": return torch.bfloat16
    if mode == "fp16": return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def train_one_epoch(cfg, loader, model, criterion, optimizer, scaler, amp_dtype, device, is_main, sampler=None):
    if sampler is not None:
        sampler.set_epoch(cfg['epoch'])
    model.train()
    sum_loss = 0.0; sum_iou = 0.0; n_samples = 0
    pbar = tqdm(total=len(loader)) if is_main else None
    autocast_enabled = amp_dtype is not None
    
    # [QAT] NNCF 컨트롤러가 있다면 스케줄러 스텝
    controller = getattr(model, 'controller', None)
    if controller:
        controller.scheduler.step()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
        else:
            out = model(x)
            loss = criterion(out, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iou, dice, _ = iou_score(out, y)
        bs = x.size(0)
        sum_loss += float(loss.item()) * bs
        sum_iou  += float(iou) * bs
        n_samples += bs

        if pbar:
            pbar.set_postfix(OrderedDict(loss=sum_loss/max(n_samples,1), iou=sum_iou/max(n_samples,1)))
            pbar.update(1)
    if pbar: pbar.close()

    stats = {'loss_sum': sum_loss, 'iou_sum': sum_iou, 'n': n_samples}
    stats = ddp_allreduce_mean(stats, device)
    tr_loss = stats['loss_sum'] / max(stats['n'], 1)
    tr_iou  = stats['iou_sum']  / max(stats['n'], 1)
    return OrderedDict(loss=tr_loss, iou=tr_iou)

@torch.no_grad()
def validate_one_epoch(cfg, loader, model, criterion, amp_dtype, device, is_main):
    # (train.py와 동일)
    model.eval()
    sum_loss = 0.0; sum_iou = 0.0; sum_dice = 0.0; n_samples = 0
    pbar = tqdm(total=len(loader)) if is_main else None
    autocast_enabled = amp_dtype is not None

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
        else:
            out = model(x)
            loss = criterion(out, y)

        iou, dice, _ = iou_score(out, y)
        bs = x.size(0)
        sum_loss += float(loss.item()) * bs
        sum_iou  += float(iou) * bs
        sum_dice += float(dice) * bs
        n_samples += bs
        if pbar:
            pbar.set_postfix(OrderedDict(loss=sum_loss/max(n_samples,1),
                                         iou=sum_iou/max(n_samples,1),
                                         dice=sum_dice/max(n_samples,1)))
            pbar.update(1)
    if pbar: pbar.close()

    stats = {'loss_sum': sum_loss, 'iou_sum': sum_iou, 'dice_sum': sum_dice, 'n': n_samples}
    stats = ddp_allreduce_mean(stats, device)
    va_loss = stats['loss_sum'] / max(stats['n'], 1)
    va_iou  = stats['iou_sum']  / max(stats['n'], 1)
    va_dice = stats['dice_sum'] / max(stats['n'], 1)
    return OrderedDict(loss=va_loss, iou=va_iou, dice=va_dice)

# ------------------------- Args -------------------------
# (기존 train.py와 동일: parse_args)
# ...
def parse_args():
    p = argparse.ArgumentParser()
    # (train.py의 모든 argparse 내용)
    # ...
    # basics
    p.add_argument('--name', default=None)
    p.add_argument('--epochs', default=300, type=int)
    p.add_argument('-b', '--batch_size', default=16, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--output_dir', default='outputs')

    # data
    p.add_argument('--dataset', default='busi')
    p.add_argument('--image_dir', required=True)
    p.add_argument('--mask_dir',  required=True)
    p.add_argument('--splits_final', type=str, required=True)

    # model
    p.add_argument('--arch', default='UKAN')
    p.add_argument('--deep_supervision', default=False, type=str2bool)
    p.add_argument('--input_channels', default=3, type=int)
    p.add_argument('--num_classes', default=1, type=int)
    p.add_argument('--input_w', default=1024, type=int)
    p.add_argument('--input_h', default=1024, type=int)
    p.add_argument('--input_list', type=list_type, default=[128,160,256])
    p.add_argument('--no_kan', action='store_true')

    # loss
    LOSS_NAMES = losses.__all__ + ['BCEWithLogitsLoss']
    p.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # optim
    p.add_argument('--optimizer', default='Adam', choices=['Adam','SGD'])
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--nesterov', default=False, type=str2bool)
    p.add_argument('--kan_lr', default=1e-2, type=float)
    p.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # scheduler
    p.add_argument('--scheduler', default='CosineAnnealingLR',
                  choices=['CosineAnnealingLR','ReduceLROnPlateau','MultiStepLR','ConstantLR'])
    p.add_argument('--min_lr', default=1e-5, type=float)
    p.add_argument('--factor', default=0.1, type=float)
    p.add_argument('--patience', default=2, type=int)
    p.add_argument('--milestones', default='1,2', type=str)
    p.add_argument('--gamma', default=2/3, type=float)
    p.add_argument('--early_stopping', default=-1, type=int)

    # resume
    p.add_argument('--resume', type=str, default='',
                  help='(QAT) fine-tuning을 시작할 FP32 체크포인트(.pth) 경로')
    p.add_argument('--resume_strict', type=str2bool, default=True)
    # [QAT] QAT fine-tuning 시, optimizer/scheduler는 새로 시작하는 것을 권장
    p.add_argument('--resume_optim', type=str2bool, default=False)
    p.add_argument('--resume_sched', type=str2bool, default=False)

    # DDP/AMP
    p.add_argument('--ddp_backend', default='nccl', choices=['nccl','gloo','mpi'])
    p.add_argument('--bucket_cap_mb', type=int, default=None)
    p.add_argument('--amp_dtype', default='auto', choices=['auto','bf16','fp16','off'])
    return p.parse_args()


# ------------------------- Main -------------------------

def main():
    seed_all()
    cfg = vars(parse_args())

    # DDP init (no-op on single GPU)
    distributed, rank, local_rank, world_size = init_distributed(cfg['ddp_backend'])
    is_main = (rank == 0)

    # 실험 폴더
    if cfg['name'] is None:
        cfg['name'] = f"{cfg['dataset']}_{cfg['arch']}_{'wDS' if cfg['deep_supervision'] else 'woDS'}"
    # [QAT] QAT용 별도 폴더
    save_dir = os.path.join(cfg['output_dir'], f"{cfg['name']}_QAT") 
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config_qat.yml'), 'w') as f:
            yaml.dump(cfg, f)
    if distributed: dist.barrier()

    tb = SummaryWriter(save_dir) if is_main else None

    # 모델/손실
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = archs.__dict__[cfg['arch']](
        cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
        embed_dims=cfg['input_list'], no_kan=cfg['no_kan']
    )
    criterion = build_criterion(cfg['loss'])

    # [수정] 데이터 로더를 먼저 생성 (캘리브레이션에 val_loader 사용)
    train_loader, val_loader, train_sampler, val_sampler = make_dataloaders(cfg, distributed)

    # --------- Resume (NNCF 래핑 전에 FP32 가중치 로드) ---------
    start_epoch, best_iou, best_dice = 0, 0.0, 0.0
    nncf_compression_state = None
    if cfg['resume']:
        start_epoch, best_iou, best_dice = load_ckpt(
            cfg['resume'],
            model,
            optimizer=None, # QAT fine-tuning 시 옵티마이저 새로 생성
            scheduler=None, # QAT fine-tuning 시 스케줄러 새로 생성
            strict=cfg['resume_strict'],
            nncf_compression_state=nncf_compression_state # QAT->QAT 재개 시
        )
        if is_main:
            print(f"=> Resumed FP32 weights from {cfg['resume']} for QAT fine-tuning")
    
    model.to(device) # NNCF 적용 전 모델을 디바이스로 이동

    # --- [QAT] NNCF QAT 설정 ---
    # 1. NNCF Config 정의
    nncf_config_dict = {
        "input_info": {
            # 배치 크기는 1로 고정 (DDP 래핑 전이므로)
            "sample_size": [1, cfg['input_channels'], cfg['input_h'], cfg['input_w']]
        },
        "compression": {
            "algorithm": "quantization",
            "preset": "performance", # "performance" (INT8) 또는 "mixed" (정확도)
            "initializer": {
                "range": {"num_init_samples": 100}, # 캘리브레이션 샘플 수
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 100}
            }
        },
        "log_dir": save_dir
    }
    
    # 2. 캘리브레이션 데이터셋 준비 (nncf.Dataset)
    def transform_fn(data_item):
        # val_loader에서 (x, y, _) 튜플이 넘어옴
        images, _, _ = data_item
        return images.to(device) # 캘리브레이션은 GPU에서 수행

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)

    # 3. NNCF Config 객체 생성
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

    # 4. QAT 모델 생성 (FP32 모델 래핑)
    if is_main:
        print("[NNCF] Creating QAT model and running calibration...")
        
    controller, model = nncf.create_compression_controller(
        model,
        config=nncf_config,
        calibration_dataset=calibration_dataset
    )
    
    # NNCF QAT 체크포인트 재개를 위한 압축 상태 저장
    nncf_compression_state = controller.get_compression_state()
    if is_main:
        print("[NNCF] QAT model created successfully.")
    # ---------------------------

    # [수정] 옵티마이저/스케줄러를 NNCF 래핑 *후* 생성
    # (QAT는 양자화 파라미터도 함께 학습해야 함)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    
    # [수정] QAT용 체크포인트 로드 (옵티/스케줄러 포함)
    # (QAT 학습을 이어서 할 경우)
    if cfg['resume'] and cfg['resume_optim'] and cfg['resume_sched']:
        print("[NNCF] Resuming optimizer and scheduler state for QAT...")
        load_ckpt(
            cfg['resume'],
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=cfg['resume_strict'],
            nncf_compression_state=nncf_compression_state
        )

    # DDP 래핑
    if distributed:
        ddp_kwargs = {}
        if cfg['bucket_cap_mb'] is not None:
            ddp_kwargs['bucket_cap_mb'] = cfg['bucket_cap_mb']
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
        # [QAT] DDP 래핑 후 NNCF 컨트롤러 브로드캐스트
        if controller:
            controller.distributed()

    # AMP 설정
    amp_dtype = select_amp_dtype(cfg['amp_dtype'])
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # --------- Train Loop (QAT Fine-tuning) ---------
    log = OrderedDict(epoch=[], lr=[], loss=[], iou=[], val_loss=[], val_iou=[], val_dice=[])
    trigger = 0

    # [QAT] QAT 학습은 epochs를 짧게 설정하는 것이 일반적입니다. (예: 10~20 에폭)
    # (argparse에서 --epochs를 조절하여 사용하세요)
    if is_main:
        print(f"[QAT] Starting QAT fine-tuning for {cfg['epochs']} epochs...")
        
    for epoch in range(start_epoch, cfg['epochs']):
        cfg['epoch'] = epoch
        if is_main:
            print(f"Epoch [{epoch}/{cfg['epochs']}]")
        
        # [QAT] NNCF 컨트롤러가 있다면 에폭 스케줄러 스텝
        if controller:
            controller.scheduler.epoch_step()

        tr = train_one_epoch(cfg, train_loader, model, criterion, optimizer,
                             scaler, amp_dtype, device, is_main, sampler=train_sampler)
        va = validate_one_epoch(cfg, val_loader, model, criterion, amp_dtype, device, is_main)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va['loss'])
            else:
                scheduler.step()

        if is_main:
            # (로그 기록 로직 동일)
            # ...
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            log['epoch'].append(epoch); log['lr'].append(current_lrs)
            log['loss'].append(tr['loss']); log['iou'].append(tr['iou'])
            log['val_loss'].append(va['loss']); log['val_iou'].append(va['iou']); log['val_dice'].append(va['dice'])
            pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log_qat.csv'), index=False)

            if tb is not None:
                # ... (tensorboard 로깅 동일)
                tb.add_scalar('train/loss', tr['loss'], epoch)
                tb.add_scalar('val/iou',    va['iou'],  epoch)
                
            save_ckpt(os.path.join(save_dir, 'last_qat.pth'),
                      model, optimizer, scheduler, epoch, best_iou, best_dice, cfg)

        if is_main and va['iou'] > best_iou:
            best_iou, best_dice, trigger = va['iou'], va['dice'], 0
            save_ckpt(os.path.join(save_dir, 'best_qat.pth'),
                      model, optimizer, scheduler, epoch, best_iou, best_dice, cfg)
            print("=> saved BEST QAT checkpoint | IoU=%.4f | Dice=%.4f" % (best_iou, best_dice))
        else:
            trigger += 1

        if cfg['early_stopping'] >= 0 and trigger >= cfg['early_stopping']:
            if is_main: print("=> early stopping")
            break

    if tb is not None:
        tb.close()

    # --- [QAT] 학습 완료 후 INT8 모델로 변환 ---
    if is_main:
        print("[QAT] Fine-tuning finished. Exporting to OpenVINO INT8...")
        try:
            # DDP 래핑 해제
            model_to_export = model.module if hasattr(model, "module") else model
            model_to_export.eval()
            
            # 더미 입력 생성
            dummy_input = torch.randn(
                1, cfg['input_channels'], cfg['input_h'], cfg['input_w']
            ).to(device)

            # OpenVINO로 변환 (QAT 모델은 자동으로 INT8로 변환됨)
            ov_model = ov.convert_model(model_to_export, example_input=dummy_input)
            
            # 최종 INT8 모델 저장
            int8_xml_path = os.path.join(save_dir, "model_qat_int8.xml")
            ov.save_model(ov_model, int8_xml_path)
            
            print(f"\n[성공] OpenVINO INT8 (QAT) 모델 저장 완료:")
            print(f"{int8_xml_path}")
            print(f"{os.path.splitext(int8_xml_path)[0]}.bin")

        except Exception as e:
            print(f"\n[오류] OpenVINO INT8 모델 변환 중 실패: {e}")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()