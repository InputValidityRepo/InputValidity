import os
# import models.densenet as dn
# import models.wideresnet as wn


import torch

def get_model(args, num_classes, load_ckpt=True, load_epoch=None):
    if args.in_dataset == 'MNIST':
        print('load model MNIST_sinvad')
        if args.model_arch == 'sinvad_CNN':
            from models.mnist import MnistClassifier
            model = MnistClassifier(img_size=28*28*1)
            print('Loading ckpt for MNIST...')
            model.load_state_dict(torch.load('。/SINVAD/sa/models/MNIST_conv_classifier_atlayer.pth'))
        if args.model_arch == 'sinvad_CNN-supcon':
            from models.mnist import MnistClassifier
            model = MnistClassifier(img_size=28 * 28 * 1)
    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {} (include head params)'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
