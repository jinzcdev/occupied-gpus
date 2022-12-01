gpu_ids="all"
option=0

if [ -n "$1" ]; then
    gpu_ids=$1
fi

if [ -n "$2" ]; then
    option=$2
fi

CUDA_DEVICE_ORDER="PCI_BUS_ID" \
python ./occupiedgpus/core.py --batch-size 32 --lr 1e-4 --eval-interval 3 --save-root ./runs --gpu-ids $gpu_ids  --option $option
