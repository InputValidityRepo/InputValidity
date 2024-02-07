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
from pcnnpp import config_udacity as config
from pcnnpp.utils.functions import get_hitmap_function
from pcnnpp.utils.Udacity_data_utils import load_test_data
rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5
default_transform = transforms.Compose([transforms.ToTensor(), rescaling]) #/255 & rescaling (-1,1)


def input_valid_driving(threshold):
    input_shape = (3,100,100)
    hitmap_function = get_hitmap_function(input_shape)
    # setting up tensorboard data summerizer
    # writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.model_name))
    # initializing model
    model = init_model(input_shape, config)
    #for IV in ['DX_occl', 'DX_blackout', 'DX_light', 'DLFuzz', 'sinvad']:
    for IV in ['sinvad']:
        if IV == 'DX_occl':
            gen_img_folder = './occl_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif IV == 'DX_blackout':
            gen_img_folder = './blackout_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif IV == 'DX_light':
            gen_img_folder = './light_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif IV == 'sinvad':
            gen_img_file = './bound_imgs_driving_torch_convVAE.npy'
            gen_img_folder = gen_img_file
            img_paths = np.load(gen_img_file)[:100]
        valid_count = 0
        scores = []
        with open('pixelcnn_driving_{0}_test_suite.txt'.format(IV), 'w') as f:
            for idx, img_path in enumerate(img_paths):
                if IV != 'sinvad':
                    img = imageio.imread(gen_img_folder + '/' + img_path).astype(np.float64)  # img 0~255, (100,100,3)
                    proc_img = default_transform(img/255.)
                else:
                    proc_img = default_transform(np.transpose(img_path.reshape(3,100,100), (1,2,0)))
                inputs = proc_img.cuda().unsqueeze(0).float()
                output = model(inputs)
                factor = 1  # negative is anomaly
                hitmap = factor * hitmap_function(inputs, output).data.cpu().numpy()
                log_prob_normality = hitmap.sum(1).sum(1)
                score = log_prob_normality
                scores.append(score)
                if score < threshold:
                    print('invalid')
                else:
                    print('gen_img is valid')
                    if IV != 'sinvad':
                        f.write('valid gen_img: {0}'.format(img_path))
                    else:
                        f.write('valid gen_img: {0}'.format(idx))
                    f.write('\n')
                    valid_count += 1
            f.write('gen_img folder = {0}'.format(gen_img_folder))
            f.write('\n')
            f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(img_paths), valid_count,
                         valid_count / len(img_paths)))
        np.save('./Udacity_{0}_prob_scores.npy'.format(IV), scores)

class Udacity_data(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        return torch.from_numpy(self.imgs[item]).permute(2,0,1), torch.from_numpy(self.labels[item][np.newaxis])

def get_threshold_Udacity():
    rescaling = lambda x: (x - .5) * 2.
    test_set_scores = []
    input_shape = (3, 100, 100)
    # initializing model
    model = init_model(input_shape, config)
    hitmap_function = get_hitmap_function(input_shape)
    (test_data, test_lbl) = load_test_data()
    rescaled_test = rescaling(np.array(test_data))
    test_set = Udacity_data(rescaled_test, test_lbl)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=True)
    for (inputs, _) in tqdm(test_loader):
        inputs = inputs.cuda()
        output = model(inputs)  # tuple (img, lbl)
        factor = 1  # negative is anomaly

        hitmap = factor * hitmap_function(inputs, output).data.cpu().numpy()
        log_prob_normality = hitmap.sum(1).sum(1)
        score = log_prob_normality  # high score=normal, low score=abnormal (bs,)
        test_set_scores.append(score)
    np.save('Udacity_result/pixelcnn_test_scores.npy', np.array(test_set_scores).reshape(-1))

if __name__ == '__main__':
    threshold_driving = -89199.734
    input_valid_driving(threshold_driving)
    #get_threshold_Udacity() #run this for getting pixelcnn++ threshold for udacity
