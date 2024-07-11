set -e
config=$1


function main 
{
    CUDA_VISIBLE_DEVICES=2 python -m TalkingFace.infer \
                           --config_path `pwd`/scripts/$config \
                           --save_path `pwd`/results/ 
}

main
