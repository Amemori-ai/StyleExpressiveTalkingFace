where_am_i=`pwd | awk -F '/' '{print $NF}'`
if [[ $where_am_i != "docker" ]];
then
    echo "please run build.sh under docker folder."
    cd ./docker
fi

mkdir -p packages

packages_list=`cat requirements_aws.txt`
for package in ${packages_list[@]};
do
    aws s3 cp s3://python-packages/${package} ./packages/${package}
done

docker build -t deep_engine:pytorch-113 -f ./Dockerfile .
rm -r packages
cd ../
