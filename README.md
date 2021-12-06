# [Occupied GPU](https://github.com/jinzcdev/occupied-gpu.git)

The program used to occupy GPUs.

## Preparation

```shell
pip install -r requirements.txt
```

## Usage

To occupy the corresponding GPUs, run:

```shell
sh train.sh 0,1,2,3
```

or

```shell
chmod a+x ./train.sh
./train.sh 0,1,2,3
```

where `0,1,2,3` is the required format.

**Note:** if you want to run the code in the background, run:

```shell
nohup sh train.sh ${GPU_IDS} >> output.log 2>&1 &
```
