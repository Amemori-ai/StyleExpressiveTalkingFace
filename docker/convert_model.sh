expname=$1
decoder_path=$2
docker run  \
       -v `pwd`:/app/ \
       -v /data1:/data1 \
       -v /data1/wanghaoran/Amemori:/data1/wanghaoran/Amemori \
       -w /app \
       --rm \
       -it deep_engine:pytorch-113 \
       python ./tools/deploy.py \
       --exp_name $expname\
       --decoder_path \
       $decoder_path \
      --to_path /app/release

aws s3 sync release/$expname  s3://update-weights/model/Voice2Lip_v1.3.1/$expname
# /app/TalkingFace/ExpressiveVideoStyleGanEncoding/results/pivot_027/snapshots/100.pth \
# /data1/chenlong/0517/video/0522/2/results/man3_chenl_0521/man3_pivot_512_10000/snapshots/200.pth \
