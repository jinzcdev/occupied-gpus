gpu_ids=${1:-0}
options=${2:-0}

CUDA_DEVICE_ORDER="PCI_BUS_ID" \
python ./occupiedgpus/core.py --gpu-ids $gpu_ids --epochs 120 --options $options
