'''
Code is built on top of DeepXplore code base. 
Density objective and VAE validation is added to the original objective function.
We use DeepXplore as a baseline technique for test generation.

DeepXplore: https://github.com/peikexin9/deepxplore
'''

from __future__ import print_function
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import argparse

from keras.layers import Input
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from Model4 import Model4
from Model_sinvadCNN import Model_sinvad
from utils import *
import imageio
import numpy as np

import datetime
from tqdm import tqdm
random.seed(3)
# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('--weight_knn', help="weight hyperparm to control vae goal", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[-1, 0, 1, 2, 3], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)
args = parser.parse_args()

print("\n\n")

knn_threshold = 0

if args.weight_knn == 0:
    output_directory = './baseline_generated_inputs_Model' + str(args.target_model + 1)+'/'+(args.transformation)+'/'
else:
    output_directory = './DX_KNN' + str(args.target_model + 1)+'/'+(args.transformation)+'/' + str(knn_threshold) + '_' + str(args.weight_knn)+'/'
    
#Create directory to store generated tests
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
delete_files_from_dir(output_directory, 'png')

# Create a subdirectory inside output directory 
# for saving original seed images used for test generation
orig_directory = output_directory+'seeds/'
if not os.path.exists(orig_directory):
    os.makedirs(orig_directory)
delete_files_from_dir(orig_directory, 'png')


# input image dimensions
img_rows, img_cols = 28, 28
img_dim = img_rows * img_cols
# the data, shuffled and split between train and test sets
#(_, _), (x_test, y_test) = mnist.load_data()
import gzip
import sys
import pickle
f = gzip.open('./mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(_, _), (x_test, y_test) = data
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)
#import mkl
#import faiss
#from keras.layers import BatchNormalization
in_dataset = 'MNIST'
name = 'sinvad_CNN'
# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
if args.target_model == 3:
    model3 = Model4(input_tensor=input_tensor)
elif args.target_model == -1:
    print('loading sinvad CNN model')
    model3 = Model_sinvad(input_tensor=input_tensor)
    model3.summary()
else:
    model3 = Model3(input_tensor=input_tensor)
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(128, 192)]))# Last Layer only (only use penultimate layer feat)
root = '../DeepKNN'
cache_name = f"{root}/cache/MNIST/{in_dataset}_train_{name}_in_alllayers.npz"
id_npzfile = np.load(cache_name)
feat_log, score_log, label_log = id_npzfile['arr_0'], id_npzfile['arr_1'], id_npzfile['arr_2']
cache_name = f"{root}/cache/MNIST/{in_dataset}_val_{name}_in_alllayers.npz"
id_npzfile_val = np.load(cache_name)
feat_log_val, score_log_val, label_log_val = id_npzfile_val['arr_0'], id_npzfile_val['arr_1'], id_npzfile_val[
    'arr_2']

ftrain = prepos_feat(feat_log)  # in_dataset train feat

def euclidean_distance(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
# init coverage table, set all to False
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

if args.weight_knn == 0:
    print("*****Running baseline test....")
else:
    print("*****Running KNN+Baseline test....")
    
# ==============================================================================================
# start gen inputs

start_time = datetime.datetime.now()
seed_nums = np.random.choice(np.arange(10000), size=300, replace=False)
result_loop_index = []
result_coverage = []
loop_index = 0
total_seeds = np.arange(10000)
#for current_seed in seed_nums:
while len(result_loop_index) < 110:
    idx = np.random.choice(range(len(ftrain)), 50)
    current_seed = total_seeds[loop_index]
    print('seed num', current_seed, 'gen_num', len(result_loop_index))
    # below logic is to track number of iterations under progress
    loop_index += 1
        
    gen_img = np.expand_dims(x_test[current_seed], axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    #if not label1 == label2 == label3 and not isInvalid(gen_img, knn, knn_threshold):
    if not label1 == label2 == label3:
        print('input already causes different outputs: {}, {}, {}'.format(label1, label2, label3))

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)
        if args.target_model == 0:
            result_coverage.append(neuron_covered(model_layer_dict1)[2])
        elif args.target_model == 1:
            result_coverage.append(neuron_covered(model_layer_dict2)[2])
        elif args.target_model == 2:
            result_coverage.append(neuron_covered(model_layer_dict3)[2])
        elif args.target_model == 3:
            result_coverage.append(neuron_covered(model_layer_dict3)[2])
        elif args.target_model == -1:
            result_coverage.append(neuron_covered(model_layer_dict3)[2])

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imageio.imwrite(output_directory + 'already_differ_' + str(current_seed) + '_' + str(label1) + '_' + str(label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue
        

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 3:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == -1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])

    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])

    # KNN distance loss:
    feat1 = model3.get_layer('activation_1').output
    feat2 = model3.get_layer('activation_2').output
    feat3 = model3.get_layer('activation_3').output
    feat4 = model3.get_layer('activation_4').output
    ood_feat_log = K.concatenate([K.mean(feat1, (1,2)), K.mean(feat2, (1,2)), K.mean(feat3, (1,2)), K.mean(feat4, (1,2))], axis=-1)

    food = (ood_feat_log[:,128:] / (K.l2_normalize(ood_feat_log[:,128:], axis=-1)) + 1e-10)

    #D, I = index.search(food.numpy(), K)
    dist = [euclidean_distance(food, ftrain[i]) for i in idx]
    sum_dist = sum(dist)  # sum over all K-NN' dists, minimize sum_dist
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron) + args.weight_knn * (-1 * sum_dist) #minimize distance from training
    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # Gradient computation of loss with respect to the input
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # function to calculate the loss and grads given the input tensor
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, sum_dist, grads])

    # Running gradient ascent, once found one valid input, break and continue to next seed
    for iters in tqdm(range(args.grad_iterations)):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, sum_dist, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        # generate the new test input
        gen_img += grads_value * args.step
        gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        if not predictions1 == predictions2 == predictions3:
            # Update coverage for valid & differential behavior gen_img
            # update info: this gen_img covers how many neurons
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            # Track the seed numbers and coverage achieved for final result
            # save seed number & coverage result for valid & diff_behavior_inducing gen_img
            result_loop_index.append(loop_index)
            if args.target_model == 0: # save coverage=num_covered / total_neuron
                result_coverage.append(neuron_covered(model_layer_dict1)[2])
                import pdb; pdb.set_trace()
            elif args.target_model == 1:
                result_coverage.append(neuron_covered(model_layer_dict2)[2])
            elif args.target_model == 2:
                result_coverage.append(neuron_covered(model_layer_dict3)[2])
            elif args.target_model == 3:
                result_coverage.append(neuron_covered(model_layer_dict3)[2])
            elif args.target_model == -1:
                result_coverage.append(neuron_covered(model_layer_dict3)[2])

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imageio.imwrite(
                    output_directory + str(loop_index) + '_' + str(
                        predictions1) + '_' + str(predictions2) + '_' + str(predictions3)+'.png',
                    gen_img_deprocessed)
            imageio.imwrite(
                    orig_directory + str(loop_index) + '_' + str(
                        predictions1) + '_' + str(predictions2) + '_' + str(predictions3)+'_orig.png',
                    orig_img_deprocessed)
            break


