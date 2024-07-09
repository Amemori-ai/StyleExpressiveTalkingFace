set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=$1 python -m TalkingFace \
                           --config_path `pwd`/scripts/${exp_name}/config.yaml \
                           --save_path `pwd`/results/${exp_name}
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`

main 0
