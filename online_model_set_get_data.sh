#!/bin/bash
set -e
ROOT_PATH=/data1/chenlong/online_model_set/exp_ori/$1

exp_name=results
folder_name=$(basename "$ROOT_PATH")

echo ${folder_name}
#mkdir xxx
directory=/data1/chenlong/online_model_set/lm_train_data/datasets/${folder_name}
mkdir -p ${directory}

gt_smooth=/data1/chenlong/online_model_set/face/${folder_name}/face
# get id.pt
#ln -s ${ROOT_PATH}/${exp_name}/cache.pt ${directory}/id.pt
cp -f ${ROOT_PATH}/${exp_name}/cache.pt ${directory}/id.pt

# get face landmark
python get_landmarks.py --from_path ${gt_smooth}  \
                        --to_path ${directory}/lm3d.npy

# get id landmark
python get_id_landmarks.py --id_path ${directory}/id.pt \
                           --landmark_path ${directory}/lm3d.npy \
                           --to_path ${directory}/id_landmark.npy
# get pose pt
python merge_more2one.py ${ROOT_PATH}/${exp_name}/pose ${directory}/pose.pt

# get attribute pt
python merge_more2one.py ${ROOT_PATH}/${exp_name}/expressive ${directory}/attribute.pt

# get train/val data
# attribute
python tools/get_validate_data.py --from_path ${directory}/attribute.pt \
                                  --to_path ${directory}/ \
                                  --ratio 0.9

# landmark
python tools/get_validate_data.py --from_path ${directory}/lm3d.npy \
                                  --to_path ${directory} \
                                  --ratio 0.9
## get training scripts
mkdir -p /data1/chenlong/online_model_set/lm_train_data/scripts
cp -r scripts/online_set_template /data1/chenlong/online_model_set/lm_train_data/scripts/${folder_name}

pose_path=${directory}/pose.pt
attr_path=${directory}/attribute.pt
id_path=${directory}/id.pt
id_landmark_path=${directory}/id_landmark.npy

attr_train_path=${directory}/train_attr.pt
attr_val_path=${directory}/val_attr.pt

ldm_train_path=${directory}/train_landmarks.npy
ldm_val_path=${directory}/val_landmarks.npy

sed --expression "s@%POSE_PATH%@${pose_path}@" \
    -e "s@%ATTRIBUTE_PATH%@${attr_path}@g" \
    -e "s@%LANDMARK_VAL_PATH%@${ldm_val_path}@g" \
    -e "s@%ID_PATH%@${id_path}@g" \
    -e "s@%ID_LANDMARK_PATH%@${id_landmark_path}@g" \
    -e "s@%ATTR_TRAIN_PATH%@${attr_train_path}@g" \
    -e "s@%ATTR_VAL_PATH%@${attr_val_path}@g" \
    -e "s@%LANDMARK_TRAIN_PATH%@${ldm_train_path}@g" \
    scripts/online_set_template/config.yaml \
    > /data1/chenlong/online_model_set/lm_train_data/scripts/${folder_name}/config.yaml

# get config_test.yaml

pose_latent_path=${ROOT_PATH}/${exp_name}/pose
train_config_path=/data1/chenlong/online_model_set/lm_train_data/scripts/${folder_name}/config.yaml
train_weight_path=/data1/chenlong/online_model_set/lm_train_data/results/${folder_name}/snapshots/best.pth
pti_path=/data1/chenlong/online_model_set/exp_ori/${folder_name}/results/pti_ft_512/snapshots
video_landmark_path=${directory}/lm3d.npy
driving_images_dir=${gt_smooth}


sed --expression "s@%POSE_LATENT_PATH%@${pose_latent_path}@" \
    -e "s@%TRAIN_CONFIG_PATH%@${train_config_path}@g" \
    -e "s@%TRAIN_WEIGHT_PATH%@${train_weight_path}@g" \
    -e "s@%PTI_WEIGHT_PATH%@${pti_path}@g" \
    -e "s@%ATTRIBUTE_PATH%@${attr_path}@g" \
    -e "s@%ID_PATH%@${id_path}@g" \
    -e "s@%LANDMARK_PATH%@${video_landmark_path}@g" \
    -e "s@%ID_LANDMARK_PATH%@${id_landmark_path}@g" \
    -e "s@%GT_PATH%@${driving_images_dir}@g" \
    scripts/online_set_template/config_test.yaml \
    > /data1/chenlong/online_model_set/lm_train_data/scripts/${folder_name}/config_test.yaml
