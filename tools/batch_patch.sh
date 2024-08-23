set -e
model_list=`cat model_list.txt`
for model in ${model_list[@]};
do
    bash docker/convert_model.sh $model \
         /data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/${model}/snapshots/best.pth \
         None \
         1
done

