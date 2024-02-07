##### this code file containing DAIV input validator
#### gen_img -> DAIV -> valid/invalid?

# load gen_img:
import imageio
import os
import numpy as np
from keras.layers import Input
from Vae_MNIST_NN1 import Vae_MNIST_NN1
from utils import *
def MNIST_validate_VAE(img_path, vae_threshold):
    gen_img_path = img_path
    img = imageio.imread(gen_img_path).astype(np.float64)  # img 0~255
    img /= 255 # img 0~1
    input_shape = img[:, :, np.newaxis].shape
    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)
    vae = Vae_MNIST_NN1(input_tensor=input_tensor)
    invalid = isInvalid(img, vae, vae_threshold) #Returns true if invalid
    return invalid


def run_validation(gen_img_folder, vae_threshold):
    img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
    valid_count = 0
    for img_path in img_paths:
        full_path = gen_img_folder + '/' + img_path
        print(full_path)
        invalid = MNIST_validate_VAE(full_path, vae_threshold)
        if invalid:
            print('gen_img is invalid')
        else:
            print('gen_img is valid')
            valid_count += 1
    with open('valid_percent.txt', 'w') as f:
        f.write('gen_img path = {0}'.format(gen_img_folder))
        f.write('\n')
        f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}, thres: {3}'.format(len(img_paths), valid_count, 100* valid_count/len(img_paths), vae_threshold))
        print('total generated imgs: {0}, valid counts: {1}, % Valid: {2}, thres: {3}'.format(len(img_paths), valid_count,
                                                                                    100 * valid_count / len(img_paths), vae_threshold))


if __name__ == '__main__':
    light = '/light_test_suite/'
    occl = '/occl_test_suite/'
    dlfuzz = '/DLFuzz_test_suite/'
    blackout = '/blackout_test_suite/'
    # modify test suite for validation based on your need.
    for gen_img_folder in [light, occl, dlfuzz, blackout]:
        vae_threshold = -2708.34 #threshold for MNIST
        run_validation(gen_img_folder, vae_threshold)