# Occupied GPUs

The program used to occupy GPUs.

## Preparation

```shell
pip install -r requirements.txt
```

> Note: Please make sure that `cuda` is avaliable. 

## Usage

### Python Module

This repository has been packaged and published in [PyPI](https://pypi.org). Please run the following command to install it.

```shell
pip install occupiedgpus
```

And happy to occupy GPUs:

```python
python -m occupiedgpus.core --gpu-ids 0,1,2,3 --epochs 120 --options 0
```

where `--options` can be assigned 0 or 1 ( `0` means to occupy the GPU when it is not used, and `1` means to occupy the remaining GPU memory at any time).

### Source Code

Clone this repository to your local and activate your Python environment.

```shell
git clone https://github.com/jinzcdev/occupied-gpus.git
```

**single processing**

To occupy the corresponding GPUs in single processing, run:

```shell
sh train.sh 0,1,2,3 [option]
```

or

```shell
chmod u+x ./train.sh
./train.sh 0,1,2,3 [option]
```

where `0,1,2,3` stands for the GPU0-3 to be occpied. `[option]` can be assigned 0 or 1 as mentioned above, and the default is 0.

**multi-processing**

To occupy GPUs faster with multi-processing, run in **bash (NOT sh):**

```shell
bash ./multi_train.sh 0,1,2,3 [option] [port]
```

where `[option]` can be assigned 0 or 1 as mentioned above, and the default is 0. The default `[port]` is 54886, and when a port conflict occurs, you may change the `[port]`. 

> Note: If your pytorch version is less than 1.9.0, please replace `torchrun` with `python -m torch.distributed.launch --use_env` in [multi_train.sh](multi_train.sh).

## Run in the background

if you want to run the code in the background, run:

```shell
nohup sh train.sh ${GPU_IDS} &>> /dev/null &
or
nohup sh multi_train.sh ${GPU_IDS} &>> /dev/null &
or
nohup python -m occupiedgpus.core --gpu-ids ${GPU_IDS} --epochs 120 --options 0 &>> /dev/null &
```

> replace ${GPU_IDS} with `0,1,...`
