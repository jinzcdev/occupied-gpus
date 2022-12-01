r"""
The programming is used to occupy free memory on the corresponding GPU.

Usage:
    $ python train.py --gpu-ids 0,1,2,3 --option 0
    or
    $ python -m occupiedgpus.core --gpu-ids 0,1,2,3 --option 0

"""
import argparse
import time
from threading import Thread

import pynvml
import torch
import torch.nn as nn

pynvml.nvmlInit()


class ComputeThread(Thread):
    r"""
    `name`: the thead name.
    `is_forced`: to occupy the free memory forcefully.
    `target`: a callable object to be invoked by the `run()`.
    `args`: the argument tuple for the target invocation.
    """

    def __init__(self, name, is_forced, *args, target=None):
        super(ComputeThread, self).__init__()
        self.name = name
        self.is_forced = is_forced
        self.target = target
        self._args = args

    def run(self):
        print(f'start {self.name}')
        try:
            self.target(*self._args)  # two arguments: x, delay
        except RuntimeError as e:
            if not self.is_forced:
                print(str(e))


def get_used_free_memory(gpu_id: int):
    r"""
    `used` and `free` in bytes (B): 2^30

    return: the remaining memory of the graphics card specified in GB

    """
    GB = 1 << 30
    if gpu_id < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used // GB, mem_info.free // GB
    else:
        return -1, -1


def init_args():
    r"""
    Enter some fake training parameters such as epochs, gpu-id.
    """

    parser = argparse.ArgumentParser(description='sum the integers at the command line')

    parser.add_argument('--gpu-ids', type=str, default="all", help='gpu ids to use')
    parser.add_argument('--batch-size', type=int, default=16, help='determines how fast the gpu takes up')
    parser.add_argument('--option', default=0, type=int, help='whether to occupy the free memory forcibly')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--weights', type=str, default='./pre-trained/vit/jx_vit_base_patch16_224_in21k-e5005f0a.pth')
    parser.add_argument('--save-root', type=str, default='./runs')

    args = parser.parse_args()
    return args


class Compute(nn.Module):
    def __init__(self, thread_id, delay=3):
        super(Compute, self).__init__()
        self.thread_id = thread_id
        self.delay = delay
        self.op = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    @torch.no_grad()
    def compute_single(self, x):
        self.op(x)

    def forward(self, x):
        i = 0
        while True:
            time.sleep(self.delay)
            self.compute_single(x)
            i += 1
            if i == 100:
                print(f'Thread {self.thread_id} is running.')
                i = 0


def allocate(gids, delta_bs, is_forced=False):
    num_gpus, cnt = len(gids), 0
    is_allocated = {}
    tid = 0
    while cnt != num_gpus:
        for gid in gids:
            if not is_allocated.get(gid, False):
                used, free = get_used_free_memory(gid)
                # round down. used==0 denotes the remaining memory is less than 1 GB.
                if used != -1 and ((is_forced and free > 1) or (not is_forced and used == 0)):
                    compute = Compute(thread_id=tid, delay=3).to(f'cuda:{gid}')
                    bs = delta_bs
                    try:
                        while True:
                            x = torch.zeros([bs, 3, 224, 224], device=f'cuda:{gid}')
                            compute.compute_single(x)
                            torch.cuda.empty_cache()
                            bs += delta_bs
                    except:
                        torch.cuda.empty_cache()
                        x = torch.zeros([max(bs - delta_bs, 2), 3, 224, 224], device=f'cuda:{gid}')
                        ComputeThread(f'Thread{tid}-GPU{gid}', is_forced, x, target=compute).start()
                        tid += 1
                    if not is_forced:
                        is_allocated[gid] = True
                        cnt += 1
        if is_forced:
            time.sleep(30)
    print("all allocated")


def main():
    args = init_args()
    if args.gpu_ids in ["-1", "all", "a", ".", "..", "..."]:
        gids = list(range(pynvml.nvmlDeviceGetCount()))
    else:
        gids = list(map(int, args.gpu_ids.split(',')))
    print("GPU IDs:", *gids)
    allocate(gids, args.batch_size, args.option != 0)


main()
