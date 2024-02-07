from datasets.preprocessing import global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
import imageio
import torch
from PIL import Image
import os
import pandas as pd
from deepSVDD import DeepSVDD
device = torch.device("cuda")

# use env sp_v2
def run_validation_deepSVDD(gen_img_file, model_ckpt_path, objective, thres):
    min_max_overall = [(-0.8826567065619495, 20.108062262467364)]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                    transforms.Normalize([min_max_overall[0][0]],
                                                         [min_max_overall[0][1] - min_max_overall[0][0]])])
    valid_count = 0
    gen_imgs = np.load(gen_img_file)[:100] #we only take 100 test inputs
    print('total gen_imgs', len(gen_imgs))
    # Initialize DeepSVDD model and set neural network \phi
    net = DeepSVDD(objective, 0.1)  # objective='soft-boundary' or 'one-class'
    net.set_network('mnist_LeNet')
    net.load_model(model_path=model_ckpt_path, load_ae=True)

    with open('valid_data_SVDD_soft_bound_MNIST.txt', 'w') as f:
        f.write('total imgs = {0}'.format(len(gen_imgs)))
        f.write('\n')
        for idx, img in enumerate(gen_imgs):
            uint_img = img.reshape(28, 28) * 255
            # preprocess the gen_img:
            print('max img', uint_img.max())
            img = Image.fromarray(np.uint8(uint_img), mode='L')
            img = transform(img)
            inputs = img.to(device).unsqueeze(0)
            outputs = net.single_tests(inputs, 'cuda')
            dist = torch.sum((outputs.squeeze().data.cpu() - torch.tensor(net.c)) ** 2)
            print('dist', dist)
            scores = dist - net.R ** 2
            print('net.R', net.R)
            print('score', scores)
            if scores > thres:
                print('gen_img with idx {0} is invalid'.format(idx))
            else:
                print('gen_img with idx {0} is valid'.format(idx))
                f.write('gen_img with idx {0} is valid'.format(idx))
                f.write('\n')
                valid_count += 1
        f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(gen_imgs), valid_count,
                                                                                    valid_count / len(gen_imgs)))

def perf_measure(y_actual, y_hat):
    '''
    y_hat: IV prediction
    y_actual: human
    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==-1:
            TN += 1
        if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
            FN += 1
    acc = (TP+TN)/(TP+TN+FP+FN)
    return(TP, FP, TN, FN), acc
def compute_acc():
    sinvad_gt = pd.read_excel('./result_final.xlsx',
                              sheet_name='sinvad')

    sinvad_lbl = np.array(sinvad_gt['Q1. Classify img'])
    sinvad_lbl[sinvad_lbl != -1] = 1
    sinvad_svdd = pd.read_csv(
        './valid_data_SVDD_soft_bound_sinvad_MNIST.txt', sep='\n')
    nested = np.array(sinvad_svdd).reshape(-1)[1:-1]
    sinvad_svdd_valid_idx = np.array([nested[i].split(' ')[3] for i in range(len(nested))]).astype(int)
    svdd_sinvad_validity_array = np.ones(100) * (-1)
    svdd_sinvad_validity_array[sinvad_svdd_valid_idx] = 1
    # (TP, FP, TN, FN)
    DAIV, acc = perf_measure(sinvad_lbl, svdd_sinvad_validity_array)
    # (TP, FP, TN, FN)
    print('------SVDD (TP, FP, TN, FN)-------')
    print('SVDD', DAIV, acc)


if __name__ == '__main__':
    ckpt_path = './ckpt/deepSVDD_ckpt_soft_boundary'
    gen_img_file = './SINVAD/bound_imgs_MNIST.npy'
    #gen_img_folder = '/light_test_suite/'
    objective = 'soft-boundary'
    thres = 0.03
    run_validation_deepSVDD(gen_img_file, ckpt_path, objective, thres)
    compute_acc()