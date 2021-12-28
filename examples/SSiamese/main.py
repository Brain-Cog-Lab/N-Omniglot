import torch
import sys
from data_loader.nomniglot_pair import NOmniglotTrainSet, NOmniglotTestSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from examples.SSiamese.model import SSiamese
import time
import numpy as np
import gflags
from collections import deque
import os


if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("way", 5, "how much way one-shot learning")
    gflags.DEFINE_integer("shot", 1, "how much shot few-shot learning")
    gflags.DEFINE_string("time", 2000, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 8, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size",64, "number of batch size")
    gflags.DEFINE_float("lr", 0.0001, "learning rate")
    gflags.DEFINE_integer("show_every", 100, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 10000, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 40000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "./", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0", "gpu ids used to train")

    Flags(sys.argv)
    T = 4
    data_type = 'event'  # frequency

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("use gpu:", Flags.gpu_ids, "to train.")
    print("way:%d, shot: %d" % (Flags.way, Flags.shot))

    seed = 346
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    trainSet = NOmniglotTrainSet(root='../../data/', use_frame=True, frames_num=T, data_type=data_type,
                                 use_npz=True, resize=105)
    testSet = NOmniglotTestSet(root='../../data/', time=Flags.time, way=Flags.way, shot=Flags.shot, use_frame=True,
                            frames_num=T, data_type=data_type, use_npz=True, resize=105)
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    testLoader = DataLoader(testSet, batch_size=Flags.way * Flags.shot, shuffle=False, num_workers=Flags.workers)

    acc_list = []
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = SSiamese(device=device)

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    best = 0
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        img1 = img1.permute(1, 0, 2, 3, 4)[:, :, 0, :, :].unsqueeze(2)
        img2 = img2.permute(1, 0, 2, 3, 4)[:, :, 0, :, :].unsqueeze(2)
        max = torch.max(img1.max(), img2.max())
        torch.cuda.empty_cache()
        if batch_id > Flags.max_iter:
            break
        optimizer.zero_grad()
        output = net.forward(img1.to(device), img2.to(device), batch_size=Flags.batch_size, time_window=T)
        loss = loss_fn(output, label.to(device))
        loss_val += loss.cpu().item()
        loss.backward()
        optimizer.step()
        if batch_id % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        # if batch_id % Flags.save_every == 0:
        #     torch.save(net.state_dict(), Flags.model_path + '/model'+ ".pt")
        if batch_id % Flags.test_every == 0  or (batch_id > 30000 and batch_id % 200 == 0):
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                test1 = test1.permute(1, 0, 2, 3, 4)[:, :, 0, :, :].unsqueeze(2)
                test2 = test2.permute(1, 0, 2, 3, 4)[:, :, 0, :, :].unsqueeze(2)
                if Flags.cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                with torch.no_grad():
                    output = net.forward(test1, test2, batch_size=Flags.way*Flags.shot, time_window=T).data.cpu().numpy()
                pred = np.argmax(output)
                if pred < Flags.shot:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right*1.0/(right+error))

            if queue[-1] > best and queue[-1]!= 1.0:
                best = queue[-1]
                # torch.save(net.state_dict(), Flags.model_path + '/model' + ".pt")
            print("data_type: %s, %d-way %d-shot, gpu %s, best:%.4f" % (
                data_type, Flags.way, Flags.shot, Flags.gpu_ids, best))

        acc = 0.0
        for d in queue:
            acc += d
        print("#"*70)
        print("final accuracy: ", acc/20)
        print("besr accuracy:", best)

