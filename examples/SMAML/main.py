import torch
import numpy as np
from data_loader.nomniglot_nw_ks import NOmniglotNWayKShot
import argparse
from torch.utils.data import Dataset, DataLoader
from examples.SMAML.meta import Meta


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            if param_group['lr']>0.005:
                param_group['lr']*=0.5
    return optimizer


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda')
    maml = Meta(args).to(device)

    best = 0
    besttest = 0

    db_train = NOmniglotNWayKShot(r'../../data/',
                             n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry,
                             train=True,
                             frames_num=args.frames_num, 
                             data_type=args.result_type)
    db_test = NOmniglotNWayKShot(r'../../data/',
                            n_way=args.n_way,
                            k_shot=args.k_spt,
                            k_query=args.k_qry,
                            train=False,
                            frames_num=args.frames_num, 
                            data_type=args.result_type)
    print(len(db_train))
    dataloadertrain = DataLoader(db_train, batch_size=args.task_num, shuffle=True, num_workers=4, pin_memory=True)
    dataloadertest = DataLoader(db_test, batch_size=args.task_num, shuffle=False, num_workers=4, pin_memory=True)
    for step in range(args.epoch):
        acctrains = []
        for x_spt, y_spt, x_qry, y_qry in dataloadertrain:
            x_spt, y_spt, x_qry, y_qry = (x_spt).to(device), y_spt.long().to(device), \
                                         (x_qry).to(device), y_qry.long().to(device)

            acctrains.append(maml(x_spt, y_spt, x_qry, y_qry)[0])

        acctrains = np.array(acctrains).mean(axis=0).astype(np.float16)
        if best < acctrains: best = acctrains

        print('step:', step, '\ttraining acc:', acctrains, "best", best,args.n_way,args.k_spt,args.frames_num,args.result_type,args.seed,"3layer")

        accstest = []
        for x_spt, y_spt, x_qry, y_qry in dataloadertest:
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.long().to(device), \
                                         x_qry.to(device), y_qry.long().to(device)

            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accstest.append(test_acc[-1])

        accstest = np.array(accstest).mean(axis=0).astype(np.float16)
        if besttest < accstest: besttest = accstest

        print('Test acc:', accstest, "best", besttest)
        db_train.reset()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=400000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--seed', type=int, help='seed', default=0)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--frames_num', type=int, help='frames_num', default=4)
    argparser.add_argument('--result_type', type=str, help='result_type', default="event")

    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=8)

    args = argparser.parse_args()

    main(args)
