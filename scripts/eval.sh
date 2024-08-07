set -e 

expname=$1

#
# config:
# 
#

CUDA_VISIBLE_DEVICES=6 \
python validate.py \
       --config_path  `pwd`/scripts/${expname}/config.yaml \
       --offset_weight_path `pwd`/results/${expname}/snapshots/best.pth \
       --pti_weight_path /data1/wanghaoran/Amemori/StyleExpressiveTalkingFace/TalkingFace/ExpressiveVideoStyleGanEncoding/results/pivot_027/snapshots/ \
       --res_path `pwd`/results/${expname}.npy
