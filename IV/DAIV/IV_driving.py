import numpy as np
from keras import backend as K
import math
from data_utils import load_test_data, preprocess
from Vae_driving import Vae_driving
from keras.layers import Input
from tqdm import tqdm
import os
from skimage.transform import resize
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def _reconstruction_probability(decoder, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """

    L = 1000
    start_index = 0
    stop_index = 1000
    if X.shape[0] < 1000:
        stop_index = X.shape[0]
    result = np.array([])
    for i in range(math.ceil(X.shape[0] / 1000)):
        reconstructed_prob = np.zeros((stop_index - start_index,), dtype='float32')
        z_mean_part = z_mean[start_index:stop_index, :]
        z_log_var_part = z_log_var[start_index:stop_index, :]
        for l in range(L):
            sampled_zs = sampling([z_mean_part, z_log_var_part])
            mu_hat, log_sigma_hat = decoder.predict(sampled_zs, steps=1)
            log_sigma_hat = np.float64(log_sigma_hat)
            sigma_hat = np.exp(log_sigma_hat) + 0.00001

            loss_a = np.log(2 * np.pi * sigma_hat)
            loss_m = np.square(mu_hat - X[start_index:stop_index, :, :, :]) / sigma_hat
            reconstructed_prob += -0.5 * np.sum(loss_a + loss_m, axis=(-1, -2, -3))
        reconstructed_prob /= L
        result = np.concatenate((result, reconstructed_prob))
        start_index += 1000
        if (stop_index + 1000) < X.shape[0]:
            stop_index += 1000
        else:
            stop_index = X.shape[0]
    return result


def reconstruction_probability(x_target, vae):
    z_mean, z_log_var, _ = vae.get_layer('encoder').predict(x_target, batch_size=128)
    reconstructed_prob_x_target = _reconstruction_probability(vae.get_layer('decoder'), z_mean, z_log_var,
                                                                   x_target)
    return reconstructed_prob_x_target


# checks whether a test input is valid or invalid
# Returns true if invalid
def isInvalid(gen_img, vae, vae_threshold):
    gen_img_density = reconstruction_probability(gen_img, vae)
    if gen_img_density < vae_threshold or math.isnan(gen_img_density):
        return True
    else:
        return False

image_size = 104
image_chn = 3

def test_input_validation(threshold):
    input_shape = (104, 104, 3)
    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)
    vae = Vae_driving(input_tensor=input_tensor, train=False, weighted_kl=False)
    # input to VAE is in (0,1) with shape (104, 104)
    # modify the path based accordingly
    for TIG in ['DX_occl_SVDD', 'DX_light_SVDD', 'DX_blackout_SVDD']:
    #for TIG in ['DLFuzz']:
        if TIG == 'DX_blackout_SVDD':
            folder_path = './Driving/DX_SVDD_driving0/blackout/0.1'
        elif TIG == 'DX_occl_SVDD':
            folder_path = './Driving/DX_SVDD_driving0/occl/0.1'
        elif TIG == 'DX_light_SVDD':
            folder_path = './Driving/DX_SVDD_driving0/light/0.1'
        elif TIG == 'DLFuzz_SVDD':
            folder_path = './DLFuzz/Udacity/test_inputs_DLFuzz_SVDD/Udacity_0.1'

        img_paths = [img for img in os.listdir(folder_path) if img.endswith(".png")]
        img_paths.sort()
        test_inputs = [preprocess(folder_path + '/' + img_path, (104, 104)) for img_path in tqdm(img_paths)]
        x_test = np.reshape(test_inputs, [-1, image_size, image_size, image_chn])
        valid_count = 0
        with open('valid_data_DAIV_{0}_test_suite_densities_full.txt'.format(TIG), 'w') as f:
            f.write('total imgs = {0}'.format(len(img_paths)))
            f.write('\n')
            for idx, img in tqdm(enumerate(x_test)):
                density = reconstruction_probability(img[np.newaxis, :, :, :], vae)
                print(density)
                if density < threshold or math.isnan(density):
                    print('invalid')
                    f.write('invalid gen_img: {0}, density = {1}'.format(img_paths[idx], density))
                    f.write('\n')
                else:
                    print('valid')
                    f.write('valid gen_img: {0}, density = {1}'.format(img_paths[idx], density))
                    f.write('\n')
                    valid_count += 1
            f.write('gen_img folder = {0}'.format(folder_path))
            f.write('\n')
            f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(img_paths), valid_count,
                                                                                        valid_count / len(img_paths)))
            print('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(img_paths), valid_count,
                                                                                        valid_count / len(img_paths)))
        f.close()

def test_input_valid_sinvad(threshold):
    input_shape = (104, 104, 3)
    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)
    vae = Vae_driving(input_tensor=input_tensor, train=False, weighted_kl=False)
    #modify the path based on your own location
    folder_path = './driving_convVAE/bound_imgs_driving_torch_convVAE.npy'
    imgs_01 = np.load(folder_path)[:100]
    imgs_01 = np.transpose(imgs_01, (0,2,3,1))
    imgs_uint8 = (imgs_01 * 255).astype(np.uint8)
    #then use nearest to resize img to (104,104)
    resized_imgs = [resize(img, (104,104), order=0, preserve_range=True) for img in imgs_uint8]
    res = np.array(resized_imgs)/255.
    valid_count = 0
    with open('valid_data_DAIV_sinvad_test_suite_densities_full.txt', 'w') as f:
        f.write('threshold at {0}'.format(threshold))
        for idx, img in tqdm(enumerate(res)):
            density = reconstruction_probability(img[np.newaxis, :, :, :], vae)
            print(density)
            if density < threshold or math.isnan(density):
                print('invalid')
                f.write('invalid gen_img: {0}, density = {1}'.format(idx, density))
                f.write('\n')
            else:
                print('valid')
                f.write('valid gen_img: {0}, density = {1}'.format(idx, density))
                f.write('\n')
                valid_count += 1
        f.write('gen_img folder = {0}'.format(folder_path))
        f.write('\n')
        f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(imgs_01), valid_count,
                                                                                    valid_count / len(imgs_01)))
    f.close()

if __name__ == '__main__':
    threshold = 30348.055 #threshold computed at full data for Udacity driving.
    test_input_validation(threshold)

