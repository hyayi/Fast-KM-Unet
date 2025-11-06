python /workspace/my/KM_UNet/inference_INT8.py --name ngtube_UKAN_multi \
                                                 --output_dir /data/image/project/ng_tube/kmnet/result \
                                                 --test_image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/imagesVal \
                                                 --test_mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/labelsVal

# python /workspace/my/KM_UNet/inference_INT8.py --name ngtube_UKAN_multi \
#                                                  --output_dir /data/image/project/ng_tube/kmnet/result \
#                                                  --test_image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/250616_chungang_clinical_dataset_images \
#                                                  --test_mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/250616_chungang_clinical_dataset_labels