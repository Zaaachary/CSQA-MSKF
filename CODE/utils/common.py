import os
import json

import torch
import random
import numpy as np

def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
            
def get_device(gpu_ids):
    if gpu_ids[0] == -1:
        device_name = 'cpu'
        print('device is cpu')
    else:
        device_name = 'cuda:' + str(gpu_ids[0])
        n_gpu = torch.cuda.device_count()
        print('device is cuda, # cuda is: %d' % n_gpu)
    device = torch.device(device_name)
    return device

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

def result_dump(args, target, file_name, folder=''):
    
    mkdir_if_notexist(os.path.join(args.result_dir, folder, file_name))
    with open(os.path.join(args.result_dir, folder, file_name), 'w', encoding='utf-8') as f:
        json.dump(target, f, ensure_ascii=False, indent=4)


class AvgVar:
    """
    维护一个累加求平均的变量
    """
    def __init__(self):
        self.var = 0
        self.steps = 0

    def inc(self, v, step=1):
        self.var += v
        self.steps += step

    def avg(self):
        return self.var / self.steps if self.steps else 0

    def __str__(self) -> str:
        return f"var:{self.var}, steps:{self.steps}"

    def __repr__(self):
        return self.__str__()

class Vn:
    """
    维护n个累加求平均的变量
    """
    def __init__(self, n):
        self.n = n
        self.vs = [AvgVar() for i in range(n)]

    def __getitem__(self, key):
        return self.vs[key]

    def init(self):
        self.vs = [AvgVar() for i in range(self.n)]

    def inc(self, vs):
        for v, _v in zip(self.vs, vs):
            v.inc(_v)

    def avg(self):
        return [v.avg() for v in self.vs]

    def list(self):
        return [v.var for v in self.vs]

    def __str__(self):
        return f"{self.n}, {str(self.vs)}"

    def __repr__(self):
        return self.__str__()
