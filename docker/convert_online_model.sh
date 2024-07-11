expname=$1
decoder_path=/data1/chenlong/online_model_set/exp_ori/${expname}/results/pti_ft_512/snapshots/100.pth
echo ${expname}
echo ${decoder_path}
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
      --to_path /app/release \
      --pwd /data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding

cp /data1/chenlong/0517/public_model/config.yaml /data1/chenlong/online_model_set/release/$expname/config.yaml
#aws s3 sync s3://update-weights/model/Voice2Lip_v1.3.1/$expname/templates  /data1/chenlong/online_model_set/release/$expname/templates
aws s3 sync /data1/chenlong/online_model_set/release/$expname  s3://update-weights/model/Voice2Lip_v1.3.1/${expname}_set_v1