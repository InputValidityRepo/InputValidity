#use env sp_v2_new
import faulthandler
faulthandler.enable()
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import os
import imageio
from tqdm import tqdm
from pcnnpp.model import init_model
from pcnnpp import config_MNIST as config
from pcnnpp.data import DatasetSelection
from pcnnpp.utils.functions import get_hitmap_function
#from pcnnpp.utils.Udacity_data_utils import load_test_data
import pandas as pd
rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5
default_transform = transforms.Compose([transforms.ToTensor(), rescaling]) #/255 & rescaling (-1,1)

def input_valid_mnist(threshold):
    root = './MNIST/deepxplore_generated_inputs_Model_sinvad0'
    input_shape = (1,28,28)
    # initializing model
    model = init_model(input_shape, config)
    hitmap_function = get_hitmap_function((1,28,28))
    #for TIG in ['DX_occl', 'DX_blackout', 'DX_light', 'DLFuzz', 'Sinvad']:
    for TIG in ['DLFuzz_KNN']:
        if TIG == 'DX_occl':
            gen_img_folder = root + '/occl_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif TIG == 'DX_blackout':
            gen_img_folder = root + '/blackout_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif TIG == 'DX_light':
            gen_img_folder = root + '/light_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        valid_count = 0
        scores = []
        with open('pixelcnn_{0}_test_suite.txt'.format(TIG), 'w') as f:
            f.write('total imgs = {0}'.format(len(img_paths)))
            f.write('\n')
            for idx, img_path in enumerate(img_paths):
                if TIG != 'Sinvad':
                    img = imageio.imread(gen_img_folder + '/' + img_path).astype(np.float64)  # img 0~255

                    proc_img = default_transform(img/255.) #(-1,1)
                else:
                    proc_img = default_transform(img_path.reshape(28,28)) #(0,1)->(-1,1)
                inputs = proc_img.unsqueeze(0).float().cuda() #(1,1,28,28)
                output = model(inputs)
                factor = 1 #negative is anomaly
                hitmap = factor * hitmap_function(inputs, output).data.cpu().numpy()
                log_prob_normality = hitmap.sum(1).sum(1)
                score = log_prob_normality  # high score=normal, low score=abnormal
                if TIG != 'Sinvad':
                    print(img_path, score)
                else:
                    print(idx, score)
                scores.append(score)
                if score < threshold:
                    print('invalid')
                else:
                    print('gen_img is valid')
                    if TIG != 'Sinvad':
                        f.write('valid gen_img: {0}'.format(img_path))
                    else:
                        f.write('valid gen_img: {0}'.format(idx))
                    f.write('\n')
                    valid_count += 1

            f.write('gen_img folder = {0}'.format(gen_img_folder))
            f.write('\n')
            f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}, threshold: {3}'.format(len(img_paths), valid_count,
                                                                                       100 * valid_count / len(img_paths), threshold))
        print('total generated imgs: {0}, valid counts: {1}, % Valid: {2}, threshold: {3}'.format(len(img_paths),
                                                                                                    valid_count,
                                                                                                    100 * valid_count / len(
                                                                                                        img_paths),
                                                                                                    threshold))
def get_threshold():
    test_set_scores = []
    input_shape = (1, 28, 28)
    # initializing model
    model = init_model(input_shape, config)
    hitmap_function = get_hitmap_function((1, 28, 28))
    dataset_test = DatasetSelection(train=False, classes=tuple(range(0, 10)))
    test_loader = dataset_test.get_dataloader(batch_size=256)
    for (inputs, _) in tqdm(test_loader):
        inputs = inputs.cuda()
        output = model(inputs)  # tuple (img, lbl)
        factor = 1  # negative is anomaly

        hitmap = factor * hitmap_function(inputs, output).data.cpu().numpy() #(256,28,28)
        log_prob_normality = hitmap.sum(1).sum(1)
        score = log_prob_normality  # high score=normal, low score=abnormal (bs,)
        test_set_scores.append(score)
    np.save('MNIST_result/pixelcnn_test_scores.npy', np.array(test_set_scores).reshape(-1))


if __name__ == '__main__':
    threshold_mnist = -1300.0
    input_valid_mnist(threshold_mnist)
    #get_threshold()
