set -e

function main
{
    CUDA_VISIBLE_DEVICES=6 python -m TalkingFace.infer \
                           --config_path `pwd`/scripts/config.yaml \
                           --save_path `pwd`/results/ 
}

main
