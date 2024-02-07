'''
usage
python3 gen_diff_SVDD.py blackout 1 .1 --step=255 100 100 .25 --target_model=0 --weight_svdd=0.1, black_spots size=(3,3)
python3 gen_diff_SVDD.py light 1 0.1 --step=10 100 50 0.25 --target_model=0 --weight_svdd=0.1
python3 gen_diff_SVDD.py occl 1 0.1 --step=10 100 50 0.25 --target_model=0 --weight_svdd=0.1, occl_size=(30, 20), start_point=(0,0)
'''

from __future__ import print_function

import argparse

from scipy.misc import imsave

from driving_models import *
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('--weight_svdd', help="weight hyperparm to control vae goal", type=float)
parser.add_argument('--step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(30,20), type=tuple)

args = parser.parse_args()
if args.weight_svdd == 0:
    output_directory = './baseline_generated_inputs_Model' + str(args.target_model) + '/' + (
        args.transformation) + '/'
else:
    output_directory = './DX_SVDD_driving' + str(args.target_model) + '/' + (args.transformation) + '/' + str(args.weight_svdd) + '/'

# Create directory to store generated tests
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
delete_files_from_dir(output_directory, 'png')

orig_directory = output_directory+'seeds/'
if not os.path.exists(orig_directory):
    os.makedirs(orig_directory)
delete_files_from_dir(orig_directory, 'png')

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model2 = Dave_norminit(input_tensor=input_tensor, load_weights=True)
model3 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
normalizer = lambda x: x / (np.linalg.norm(x + 1e-10, ord=2, axis=-1, keepdims=True))
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only (only use penultimate layer feat)

feat_npz = np.load('./Udacity_testset_dim_10.npy.npz')
feat_log = feat_npz['arr_0']
ftrain = prepos_feat(feat_log)  # in_dataset train feat
ftrain_mean = ftrain.mean(0)

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
# ==============================================================================================
# start gen inputs
img_paths = image.list_pictures('./testing/center', ext='jpg')
result_loop_index = []
while len(result_loop_index) < 100:
    print('Generated input number', len(result_loop_index))
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    angle1, angle2, angle3 = model1.predict(gen_img)[0], model2.predict(gen_img)[0], model3.predict(gen_img)[0]
    if angle_diverged(angle1, angle2, angle3):
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(angle1, angle2,
                                                                                            angle3) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        #gen_img_deprocessed = draw_arrow(deprocess_image(gen_img), angle1, angle2, angle3)
        gen_img_deprocessed = deprocess_image(gen_img)
        # save the result to disk
        imsave('./generated_inputs/' + 'already_differ_' + str(angle1) + '_' + str(angle2) + '_' + str(angle3) + '.png',
               gen_img_deprocessed)
        continue

    # if all turning angles roughly the same
    orig_angle1, orig_angle2, orig_angle3 = angle1, angle2, angle3
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = K.mean(model3.get_layer('before_prediction').output[..., 0])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = K.mean(model3.get_layer('before_prediction').output[..., 0])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_prediction').output[..., 0])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])

    #get intermediate feature
    # KNN distance loss:
    feat = model1.get_layer('fc4').output
    food = (feat / (K.l2_normalize(feat + 1e-10, axis=-1)))
    dist = euclidean_distance(food, ftrain_mean)  # hypersphere center c
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron) + args.weight_svdd * (-1 * dist)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads, dist])

    # we run gradient ascent for 20 steps
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value, distance = iterate(
            [gen_img])
        print('minimize distance', distance)
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        angle1, angle2, angle3 = model1.predict(gen_img)[0], model2.predict(gen_img)[0], model3.predict(gen_img)[0]

        if angle_diverged(angle1, angle2, angle3):
            result_loop_index.append(1)
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            #gen_img_deprocessed = draw_arrow(deprocess_image(gen_img), angle1, angle2, angle3)
            #orig_img_deprocessed = draw_arrow(deprocess_image(orig_img), orig_angle1, orig_angle2, orig_angle3)
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)
            # save the result to disk
            imsave(output_directory + args.transformation + '_' + str(angle1) + '_' + str(angle2) + '_' + str(
                angle3) + '.png', gen_img_deprocessed)
            imsave(orig_directory + args.transformation + '_' + str(angle1) + '_' + str(angle2) + '_' + str(
                angle3) + '_orig.png', orig_img_deprocessed)
            break
