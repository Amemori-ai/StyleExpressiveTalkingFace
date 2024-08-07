set -e
config=$1
model_list=-1

function main 
{
    echo $1
    CUDA_VISIBLE_DEVICES=2 python -m TalkingFace.infer \
                           --config_path `pwd`/scripts/$1 \
                           --save_path `pwd`/results/ 
}

if [[ $config == *"template"* ]];
then

    if [[ "$#" -gt 1 ]];
    then
        model_list=`cat $2`
        
        for model in ${model_list[@]};
        do
            touch scripts/tmp.yaml
            echo $model
            sed -e "s/%EXPNAME%/${model}/g" $config > scripts/tmp.yaml
            main tmp.yaml
        done
    fi
else

main $config

fi


