import torch
import numpy as np
from data_loader.nomniglot_full import NOmniglotfull
import argparse
import torchvision
from examples.CSNN.meta import Meta


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    device = torch.device('cuda')
    model = Meta(args).to(device)
    TrainSet = NOmniglotfull(root=r'../../data/', train=True, frames_num=4, data_type='event',
                             transform=torchvision.transforms.Resize((28, 28)))
    TestSet = NOmniglotfull(root=r'../../data/', train=False, frames_num=4, data_type='event',
                            transform=torchvision.transforms.Resize((28, 28)))

    trainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(TestSet, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    best = 0
    besttest = 0
    for step in range(args.epoch):
        accs = []
        for x, y in trainLoader:
            x, y = x .float().to(device), y .long().to(device)
            accs.append(model(x, y))

        accs = np.array(accs).mean(axis=0).astype(np.float16)
        if best < accs: best = accs
        print('\ttraining acc:', accs, "best", best)

        accstest = []
        for x, y in testLoader:

            x, y = x .float().to(device), (y).long().to(device)

            test_acc = model.test(x, y)
            accstest.append(test_acc)

        accstest = np.array(accstest).mean(axis=0).astype(np.float16)
        if besttest < accstest: besttest = accstest
        print('Test acc:', accstest, "best", besttest)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=400000)
    argparser.add_argument('--seed', type=int, help='seed number', default=0)

    args = argparser.parse_args()

    main(args)
