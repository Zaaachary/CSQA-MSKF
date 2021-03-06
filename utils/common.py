import os
import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch

def save_csv(data, path,sep=',', type='default'):
    if len(data) < 1:
        return 
    content = ''
    if type == 'default':
        for dat in data:
            temp = sep.join(dat) + '\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(temp)
                      
def make_file(fileName):
    if fileName[-1] == '/':
        fileName = fileName + 'mk.txt'
    if not os.path.exists(fileName):
        if not os.path.isdir(fileName):
            (path, file) = os.path.split(fileName)
            if not os.path.exists(path):
                os.makedirs(path)
            try:
                fp = open(fileName, 'w')
                fp.close()
            except:
                pass

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

class F1_Measure:
    """
    ----------------
            真实
            P   N
    预   P  tp  fp
    测   N  fn  tn
    ----------------

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * prec * recall / (prec + recall)
       = 2 * tp / (tp + fp) * tp / (tp + fn) / [ tp / (tp + fp) + tp / (tp + fn)]
       = 2 * tp / [tp + fp + tp + fn]
    """
    def __init__(self):
        self.tp = 0
        self.tp_fp_tp_fn = 0

    def inc(self, tp, tp_fp, tp_fn):
        # tp_fp: 预测值为正的
        # tp_fn: 真实值为正的
        self.tp += tp
        self.tp_fp_tp_fn += tp_fp + tp_fn

    def f1(self):
        f1 = 2 * self.tp / self.tp_fp_tp_fn if self.tp else 0
        return f1


def f1_measure(tp, fp, fn):
    return 2 * tp / (tp + fp + tp + fn) if tp else 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
