set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 
parent_exp_name=`echo $bash_script | awk -F '/' '{print $2}'`

mkdir -p results

function main
{
    DEBUG=True \
    CUDA_VISIBLE_DEVICES=6 python -m TalkingFace \
                           --config_path `pwd`/scripts/${parent_exp_name}/config.yaml \
                           --save_path `pwd`/results/${parent_exp_name}/${exp_name} \
                           --resume_path `pwd`/results/${parent_exp_name}/snapshots/40.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
