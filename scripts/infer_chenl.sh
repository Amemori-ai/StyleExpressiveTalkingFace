set -e
exp_name=$2
devices=$1
function main
{
    CUDA_VISIBLE_DEVICES=${devices} python -m TalkingFace.infer \
                           --config_path `pwd`/scripts/lm_train_data/scripts/${exp_name}/config_test.yaml \
                           --save_path  /data1/chenlong/online_model_set/lm_train_data/results/
}
main
