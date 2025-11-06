python /workspace/my/KM_UNet/openvino/inference.py --name ngtube_UKAN_multi \
                                                 --output_dir /data/image/project/ng_tube/kmnet/result \
                                                 --test_image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/imagesTs \
                                                 --test_mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/labelsTs \
                                                 --model_xml /data/image/project/ng_tube/kmnet/result/ngtube_UKAN_multi/openvino_nncf_int8/quantized_model.xml

python /workspace/my/KM_UNet/openvino/inference.py --name ngtube_UKAN_multi \
                                                 --output_dir /data/image/project/ng_tube/kmnet/result \
                                                 --test_image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/250616_chungang_clinical_dataset_images \
                                                 --test_mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3004_NGT_hospitals/250616_chungang_clinical_dataset_labels \
                                                 --model_xml /data/image/project/ng_tube/kmnet/result/ngtube_UKAN_multi/openvino_nncf_int8/quantized_model.xml