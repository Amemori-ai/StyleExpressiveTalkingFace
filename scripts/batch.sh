#!/usr/bin/bash
set -e

model_list=`cat $1`

for model in ${model_list[@]};
do
    bash ./docker/convert_model.sh \
         $model \
         /data1/chenlong/online_model_set/speed/exp_speed_v3/exp/$model/v3_pti_512_ft5/snapshots/5.pth \
         /data1/chenlong/online_model_set/face/$model/face         
done








