gpu_ids="0"
options=0

if [ -n "$1" ]; then
    gpu_ids=$1
fi

if [ -n "$2" ]; then
    options=$2
fi

python train.py --gpu-ids $gpu_ids --epochs 120 --options $options
