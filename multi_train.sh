gpu_ids=${1:-"0"}  # e.g. 0,1,2
option=${2:-0}
port=${3:-54886}

arr=(${gpu_ids//,/ })
num_gpus=${#arr[@]}

CUDA_DEVICE_ORDER="PCI_BUS_ID" \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$gpu_ids \
torchrun --nproc_per_node=$num_gpus --master_port=$port ./occupiedgpus/multi_core.py \
--batch-size 8 --lr 1e-4 --weights ./runs/checkpoint.pth --eval-interval 3 --save-root ./runs \
--gpu-ids $gpu_ids  --option $option