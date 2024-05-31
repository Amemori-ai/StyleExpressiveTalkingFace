expname=$1
docker run  \
       -v `pwd`:/app/ \
       -v /data1/wanghaoran/Amemori:/data1/wanghaoran/Amemori \
       -w /app \
       --rm \
       -it deep_engine:pytorch-113 \
       python ./tools/deploy.py \
       --exp_name $expname\
       --decoder_path \
       /data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/pivot_027/snapshots/100.pth \
       --to_path /app/release

cp release/$expname -r /data1/wanghaoran/Amemori/efs_model/model/Voice2Lip_v1.3.1/$expname
