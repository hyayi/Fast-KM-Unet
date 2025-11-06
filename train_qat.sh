export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_qat.py \
    --name "my_experiment_name" \
    --output_dir "outputs" \
    --dataset "busi" \
    --image_dir "/path/to/images" \
    --mask_dir "/path/to/masks" \
    --splits_final "/path/to/splits.json" \
    --arch "UKAN" \
    --input_w 1024 \
    --input_h 1024 \
    --batch_size 8 \
    --loss "BCEDiceLoss" \
    --optimizer "Adam" \
    --amp_dtype "bf16" \
    \
    # --- QAT 핵심 인자 ---
    --resume "outputs/my_experiment_name/best.pth" \ # 1. 기존 FP32 가중치
    --epochs 15 \                                  # 2. 짧은 에폭 (미세조정)
    --lr 1e-5                                       # 3. 낮은 학습률