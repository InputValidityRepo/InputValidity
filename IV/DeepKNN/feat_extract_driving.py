#! /usr/bin/env python3
#python3 feat_extract_driving.py
import torch
from models.pilot_net import PilotNet
import os
import random
from torch.utils.data import Dataset
from util.args_loader import get_args
import numpy as np
import torch.nn.functional as F
import time
from tqdm import tqdm
from keras_preprocessing import image
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = 32
def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)/255
    return np.array(input_img_data)
def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]   #[100,100,3]

def load_train_data(path='./datasets/Ch2_002/'):
    print('start loading train data')
    xs = []
    ys = []
    start = time.time()
    a=0
    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + line.split(',')[5])
            ys.append(float(line.split(',')[6]))

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_img = []
    train_lbl = []
    for img_path, y in tqdm(zip(xs, ys)):
        processed_img = preprocess(img_path, (100,100))
        train_img.append(processed_img)
        train_lbl.append(y)

    print('finished loading training data with time', time.time() - start)
    return np.array(train_img), np.array(train_lbl)
def load_test_data(path='./Driving/testing/'):
    print('start loading test data')
    xs = []
    ys = []
    a=0
    with open(path + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_img = []
    train_lbl = []
    for img_path, y in tqdm(zip(xs, ys)):
        processed_img = preprocess(img_path, (100, 100))
        train_img.append(processed_img)
        train_lbl.append(y)
    return np.array(train_img), np.array(train_lbl)
# Driving dataset
class Udacity_data(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        return torch.from_numpy(self.imgs[item]), torch.from_numpy(self.labels[item][np.newaxis])

class Udacity_ood_data(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):
        return torch.from_numpy(self.imgs[item])
model = PilotNet()
model.cuda()
path = './end2end-self-driving-car/log/correct_ckpt/weights_final.pth'
model.load_state_dict(torch.load(path))
print("models loaded...")
model.eval()

FORCE_RUN = False

dummy_input = torch.zeros((batch_size, 100,100,3)).cuda()
score, feature_list = model.feature_list(dummy_input)
feature_list = feature_list[-1]

featdims = [10]
begin = time.time()
def save_in_feat():
    print('Begin saving IN dataset features')
    train_imgs, train_labels = load_train_data()
    test_imgs, test_labels = load_test_data()
    train_dataset, test_dataset = Udacity_data(train_imgs, train_labels), Udacity_data(test_imgs, test_labels)
    trainloaderIn = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)
    testloaderIn = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True)
    for split, in_loader in [('train', trainloaderIn), ('test', testloaderIn),]:
        cache_name = f"cache/Driving_{split}_PilotNet_dim_10.npz"
        if FORCE_RUN or not os.path.exists(cache_name):
            if split == 'train':
                length = len(train_dataset)
                feat_log = np.zeros((len(train_dataset), sum(featdims)))
                score_log = np.zeros((len(train_dataset)))
                label_log = np.zeros(len(train_dataset))
            elif split == 'test':
                length = len(test_dataset)
                feat_log = np.zeros((len(test_dataset), sum(featdims)))
                score_log = np.zeros((len(test_dataset)))
                label_log = np.zeros(len(test_dataset))
            print('length', length)
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, length)
                print(start_ind, end_ind)
                score, feature_list = model.feature_list(inputs)
                feature_list = feature_list[-2]
                out = feature_list
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.squeeze().data.cpu().numpy()
                score_log[start_ind:end_ind] = score.squeeze().data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
            np.savez(cache_name, feat_log, score_log, label_log)
        else:
            npzfile = np.load(cache_name)
            feat_log, score_log, label_log = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']

def save_out_feat():
    print('Begin saving OOD dataset features')
    out_datasets = ['DX_occl_SVDD_drive', 'DX_light_SVDD_drive', 'DX_blackout_SVDD_drive', 'DLFuzz_SVDD_drive']

    for ood_dataset in out_datasets:
        print('out ood_dtaset is', ood_dataset)
        if ood_dataset == 'DX_occl' or ood_dataset == 'DX_light' or ood_dataset == 'DX_blackout':
            gen_img_folder = '/public/home/zhangjy/anomaly/deepxplore/Driving/generated_imgs/{0}_test_suite'.format(ood_dataset.split('_')[1])
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            imgs_list = []
            print('total 100 imgs', len(img_paths))
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100,100))
                imgs_list.append(img)
        elif ood_dataset == 'DLFuzz':
            gen_img_folder = '/public/home/zhangjy/anomaly/DLFuzz/Udacity/generated_inputs/Udacity'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            imgs_list = []
            print('total 100 imgs', len(img_paths))
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100, 100))
                imgs_list.append(img)
        elif ood_dataset == 'SINVAD':
            imgs_list = []
            path = '/public/home/zhangjy/anomaly/SINVAD/results/driving/bound_imgs_driving_torch.npy'
            bound_imgs = np.load(path)[:100]   # we only take 100 test inputs
            for img in bound_imgs:  #0~1
                imgs_list.append(np.transpose(img.reshape(3,100,100), (1,2,0)))
        elif ood_dataset == 'DX_light_SVDD_drive':
            imgs_list = []
            gen_img_folder = '/home/jzhang2297/anomaly/deepxplore/Driving/DX_SVDD_driving0/light/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100, 100))
                imgs_list.append(img)
        elif ood_dataset == 'DX_blackout_SVDD_drive':
            imgs_list = []
            gen_img_folder = '/home/jzhang2297/anomaly/deepxplore/Driving/DX_SVDD_driving0/blackout/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100, 100))
                imgs_list.append(img)
        elif ood_dataset == 'DX_occl_SVDD_drive':
            imgs_list = []
            gen_img_folder = '/home/jzhang2297/anomaly/deepxplore/Driving/DX_SVDD_driving0/occl/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100, 100))
                imgs_list.append(img)
        elif ood_dataset == 'DLFuzz_SVDD_drive':
            imgs_list = []
            gen_img_folder = '/home/jzhang2297/anomaly/DLFuzz/Udacity/test_inputs_DLFuzz_SVDD/Udacity_0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
            for img_path in img_paths:
                full_path = gen_img_folder + '/' + img_path
                img = preprocess(full_path, (100, 100))
                imgs_list.append(img)
        cache_name = f"cache/{ood_dataset}vsDriving_PilotNet_out_dim_10.npz"
        if FORCE_RUN or not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(imgs_list), sum(featdims)))
            ood_score_log = np.zeros(len(imgs_list))

            model.eval()
            for batch_idx, inputs in enumerate(imgs_list):
                inputs = torch.tensor(inputs).unsqueeze(0)
                inputs = inputs.to(device)
                score, feature_list = model.feature_list(inputs.float())
                feature_list = feature_list[-1]
                out = feature_list

                ood_feat_log[batch_idx, :] = out.squeeze().data.cpu().numpy()
                ood_score_log[batch_idx] = score.squeeze().data.cpu().numpy()
                if (batch_idx+1) % 100 == 0:
                    print(f"{batch_idx}/{len(imgs_list)}")
            np.savez(cache_name, ood_feat_log, ood_score_log)
        else:
            npzfile = np.load(cache_name)
            ood_feat_log, ood_score_log = npzfile['arr_0'], npzfile['arr_1']


if __name__ == '__main__':
    #save_in_feat()
    save_out_feat()