set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results
exp_name=$2
function main
{
    CUDA_VISIBLE_DEVICES=2 python -m TalkingFace \
                           --config_path `pwd`/scripts/lm_train_data/scripts/${exp_name}/config.yaml \
                           --save_path /data1/chenlong/online_model_set/lm_train_data/results/${exp_name}
}

_timestamp=`date +%Y%m%d%H`

main 0
