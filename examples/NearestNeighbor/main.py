import torch
from data_loader.nomniglot_pair import NOmniglotTestSet
from torch.utils.data import DataLoader
import numpy as np
import gflags
import sys
import os
from tqdm import tqdm

if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("way", 5, "how much way one-shot learning")
    gflags.DEFINE_integer("shot", 1, "how much shot few-shot learning")
    gflags.DEFINE_string("time", 10000, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_string("gpu_ids", "0", "gpu ids used to train")
    Flags(sys.argv)
    T = 4
    data_type = 'event'

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("use gpu:", Flags.gpu_ids, "to train.")
    print("%d way %d shot" % (Flags.way, Flags.shot))

    testSet = NOmniglotTestSet(root='../../data/', time=Flags.time, way=Flags.way, shot=Flags.shot, use_frame=True,
                               frames_num=T,
                               data_type=data_type, use_npz=True, resize=105)
    testLoader = DataLoader(testSet, batch_size=Flags.way * Flags.shot, shuffle=False, num_workers=Flags.workers)

    right, error = 0, 0
    for _, (test1, test2) in tqdm(enumerate(testLoader, 1)):
        if Flags.cuda:
            test1, test2 = test1.cuda(), test2.cuda()
        with torch.no_grad():
            L2_dis = ((test1 - test2) ** 2).sum(1).sum(1).sum(1).sum(1).sqrt()
            pred = L2_dis.argmin().cpu().item()
            if pred < Flags.shot:
                right += 1
            else:
                error += 1
    print("#" * 70)
    print("final accuracy: {:.4f}".format(right / (right + error) * 100))
