# -*- coding: utf-8 -*-
#python gen_diff_driving.py [2] 0.5 5 Udacity 5
from __future__ import print_function
from keras.layers import Input
from scipy.misc import imsave
from utils_tmp import *
import sys
import os
import time
from keras_preprocessing import image
from DAVE_orig import Dave_orig
# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
print(model1.name)

# model_layer_dict1 = init_coverage_tables(model1)
model_layer_times1 = init_coverage_times(model1)  # times of each neuron covered
model_layer_times2 = init_coverage_times(model1)  # update when new image and adversarial images found
model_layer_value1 = init_coverage_value(model1)
# start gen inputs
img_paths = image.list_pictures('。/testing/center', ext='jpg')[100:600]
img_num = len(img_paths)
print(sys.argv) #['gen_diff_driving.py', '[2]', '0.5', '5', 'Udacity', '5']
# e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
neuron_to_cover_num = int(sys.argv[3])
subdir = sys.argv[4]
iteration_times = int(sys.argv[5]) #5
neuron_to_cover_weight = 0.5
predict_weight = 0.5
learning_step = 1

weight_svdd = 0.1

save_dir = './test_inputs_DLFuzz_SVDD/' + subdir + '_' + str(weight_svdd) + '/'

normalizer = lambda x: x / (np.linalg.norm(x + 1e-10, ord=2, axis=-1, keepdims=True))
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only (only use penultimate layer feat)

feat_npz = np.load('./Driving/DeepXplore/Udacity_testset_dim_10.npy.npz')
feat_log = feat_npz['arr_0']
ftrain = prepos_feat(feat_log)  # in_dataset train feat
ftrain_mean = ftrain.mean(0)

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0

total_perturb_adversial = 0

for i in range(img_num):
    start_time = time.clock()
    img_list = []
    img_path = img_paths[i]
    print(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    tmp_img = preprocess_image(img_path)
    orig_img = tmp_img.copy()
    img_list.append(tmp_img)
    update_coverage(tmp_img, model1, model_layer_times2, threshold)
    while len(img_list) > 0:
        gen_img = img_list[0] #(-155,155)

        img_list.remove(gen_img)

        # first check if input already induces differences
        pred1 = model1.predict(gen_img)
        orig_angle = pred1

        update_coverage_value(gen_img, model1, model_layer_value1)
        update_coverage(gen_img, model1, model_layer_times1, threshold)

        loss1 = -predict_weight * K.mean(model1.get_layer('before_prediction').output[..., 0])
        layer_output = loss1

        # neuron coverage loss
        loss_neuron = neuron_selection(model1, model_layer_times1, model_layer_value1, neuron_select_strategy,
                                       neuron_to_cover_num, threshold)
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2

        layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

        # KNN distance loss:
        feat = model1.get_layer('fc4').output
        food = (feat / (K.l2_normalize(feat + 1e-10, axis=-1)) )
        dist = euclidean_distance(food, ftrain_mean)  # hypersphere center c
        layer_output -= weight_svdd * dist

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = [loss1]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], [grads_tensor_list, dist, feat])

        # we run gradient ascent for 3 steps
        for iters in range(iteration_times):

            loss_neuron_list, distance, feat = iterate([gen_img])
            print('minimizing dist', distance)
            perturb = loss_neuron_list[-1] * learning_step

            gen_img += perturb

            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]

            pred1 = model1.predict(gen_img)
            angle1 = pred1

            update_coverage(gen_img, model1, model_layer_times1, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm / orig_L2_norm #use (0,1) scale to compute this

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)
            if angle_diverged(angle1, orig_angle):
                print('angle diverged', angle1, orig_angle)
                update_coverage(gen_img, model1, model_layer_times2, threshold)
                total_norm += L2_norm
                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()
                gen_img_deprocessed = deprocess_image(gen_img_tmp)
                save_img = save_dir + img_name + '.png'
                print('save adv_img at', save_img)
                imsave(save_img, gen_img_deprocessed)
                adversial_num += 1

    end_time = time.clock()

    print('covered neurons percentage %d neurons %.3f'
          % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    duration = end_time - start_time

    print('used time : ' + str(duration))

    total_time += duration

print('covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

print('total_time = ' + str(total_time))
print('average_norm = ' + str(total_norm / adversial_num))
print('adversial num = ' + str(adversial_num))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))

