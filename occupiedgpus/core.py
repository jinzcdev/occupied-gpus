r'''
The programming is used to occupy free video memories at the corresponding gpu_id.

$ python train.py --gpu-ids 0,1,2,3 --epochs 120 --options 0

'''
import argparse
import pynvml
import time

import torch
import torch.nn as nn

from threading import Thread

pynvml.nvmlInit()


class ComputeThread(Thread):
    '''

    分线程计算张量的主函数, 定时3秒钟计算一次

    *name* is the thead name.

    *target* is a callable object to be invoked by the `run()`.

    *args* is the argument tuple for the target invocation.

    '''

    def __init__(self, name, *args, target=None):
        super(ComputeThread, self).__init__()
        self.name = name
        self.target = target
        self._args = args

    def run(self):
        print(f'starting {self.name}')
        try:
            self.target(*self._args)  # 输入两个参数 (x, delay)
        except RuntimeError as e:
            print(str(e))


def get_used_free_memory(gpu_id: int):
    '''
    used与free的单位: 字节(B)
    2^30 = 1073741824

    return: 指定显卡的剩余显存容量, 单位GB

    '''
    if gpu_id < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used // 1073741824, mem_info.free // 1073741824
    else:
        return -1, -1


def init_args():
    '''
    初始化输入参数, 假装输入一些训练参数, 如 epochs
    '''

    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')

    parser.add_argument(
        '--gpu-ids', default='0', type=str,
        help='gpu ids to be used')

    parser.add_argument(
        '--epochs', default=1000, type=int,
        help='the number of epoch')

    parser.add_argument(
        '--options', default=0, type=int,
        help='options: whether to occupy the free video memory forcefully'
    )

    args = parser.parse_args()
    return args


class Compute(nn.Module):
    def __init__(self, thread_id=0, delay=3):
        super(Compute, self).__init__()
        self.thread_id = thread_id
        self.delay = delay

    def forward(self, x):
        i = 0
        while True:
            time.sleep(self.delay)
            for _ in range(3):
                x = x @ x @ x
            i += 1
            if i == 100:
                print(f'Thread {self.thread_id} is running.')
                i = 0


def allocate(gids, is_forced=False):
    num_gpus, cnt = len(gids), 0
    is_allocated = {}
    while cnt != num_gpus:
        for i, gid in enumerate(gids):
            if not is_allocated.get(gid, False):
                used, free = get_used_free_memory(gid)
                # 向下取整, used==0 denotes 显存使用不足1GB.
                if used != -1 and ((is_forced and free > 1) or (not is_forced and used == 0)):
                    x = torch.randn(
                        (2 * (free-1), 512*(256-2**abs(i-num_gpus//2)), 16, 16))
                    x = x.to(f'cuda:{gid}')
                    compute = Compute(thread_id=i, delay=3)
                    compute = compute.to(f'cuda:{gid}')
                    ComputeThread(f'Thread-{i}-GPU{gid}',
                                  x, target=compute).start()
                    is_allocated[gid] = True
                    cnt += 1


def main():
    args = init_args()
    try:
        gids = list(map(int, args.gpu_ids.split(',')))
        allocate(gids, args.options != 0)
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    main()
