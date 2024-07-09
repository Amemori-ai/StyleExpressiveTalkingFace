#!/bin/bash
set -e
ROOT_PATH=$1

exp_name=$2

#mkdir xxx
mkdir -p dataset/${exp_name}
directory=dataset/${exp_name}
# get id.pt
ln -s ${ROOT_PATH}/${exp_name}/cache.pt ${directory}/id.pt

# get e4e landmark
python get_landmarks.py --from_path ${ROOT_PATH}/${exp_name}/data/smooth/  \
                        --to_path ${directory}/lm3d.npy

# get id landmark
python get_id_landmarks.py --id_path ${directory}/id.pt \
                           --landmark_path ${directory}/lm3d.npy \
                           --to_path ${directory}/id_landmark.npy
# get pose pt
python merge_more2one.py ${ROOT_PATH}/pose ${directory}/pose.pt

# get attribute pt
python merge_more2one.py ${ROOT_PATH}/expressive ${directory}/attribute.pt

# get train/val data
# attribute
python tools/get_validate_data.py --from_path ${directory}/attribute.pt \
                                  --from_path ${diretory} \
                                  --ratio 0.1

# landmark
python tools/get_validate_data.py --from_path ${directory}/lm3d.npy \
                                  --from_path ${diretory} \
                                  --ratio 0.1
# get training scripts
cp -r scripts/template scripts/${exp_name}

pose_path=${directory}/pose.pt
attr_path=${directory}/attribute.pt
id_path=${directory}/id.pt
id_landmark_path=${directory}/id_landmark.npy

attr_train_path=${director}/attribute_train.pt
attr_val_path=${director}/attribute_val.pt

ldm_train_path=${director}/lm3d_e4e_train
ldm_val_path=${director}/lm3d_e4e_val.np

sed --expression "s@%POSE_PATH%@${pose_path}@" \
    -e "s@%ATTRIBUTE_PATH%@${attr_path}@g" \
    -e "s@%LANDMARK_VAL_PATH%@${ldm_val_path}@g" \
    -e "s@%ID_PATH%@${id_path}@g" \
    -e "s@%ID_LANDMARK_PATH%@${id_landmark_path}@g" \
    -e "s@%ATTR_TRAIN_PATH%@${attr_train_path}@g" \
    -e "s@%ATTR_VAL_PATH%@${attr_val_path}@g" \
    -e "s@%LANDMARK_TRAIN_PATH%@${ldm_train_path}@g" \
    scripts/template/config.yaml \
    > scripts/${exp_name}/config.yaml

