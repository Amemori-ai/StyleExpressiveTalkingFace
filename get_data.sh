set -e
exp_name=$1

#mkdir xxx
mkdir -p dataset/exp010/${exp_name}
directory=dataset/exp010/${exp_name}

# get id.pt
ln -s /data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/exp010/${exp_name}/cache.pt ${directory}/id.pt

# get e4e landmark


# get id landmark

python get_id_landmarks.py --id_path  \
                           --landmark_path \
                           --to_path
