import numpy as np
from keras.utils import load_img, img_to_array
import torch
import os
from deepSVDD_driving import DeepSVDD
device = torch.device("cuda")

def preprocess_image(img_path, target_size=(100, 100)):
    img = load_img(img_path, target_size=target_size)
    input_img_data = img_to_array(img)
    return np.array(input_img_data)

def run_validation_deepSVDD(gen_img_folder, model_ckpt_path, objective):
    valid_count = 0
    img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
    img_paths.sort()
    # Initialize DeepSVDD model and set neural network \phi
    net = DeepSVDD(objective, 0.1)  # objective='soft-boundary' or 'one-class'
    net.set_network()
    net.load_model(model_path=model_ckpt_path, load_ae=True)
    with open('valid_data_SVDD_soft_bound_DLFuzz_driving_test_suite.txt', 'w') as f:
        f.write(objective)
        f.write('\n')
        for img_path in img_paths:
        #for idx, img in enumerate(gen_imgs):
            full_path = gen_img_folder + '/' + img_path
            img = preprocess_image(full_path)
            inputs = torch.from_numpy(img).permute(2, 0, 1) / 255.
            outputs = net.single_tests(inputs, 'cuda')
            dist = torch.sum((outputs.squeeze().data.cpu() - torch.tensor(net.c)) ** 2)
            print('dist', dist)
            scores = dist - net.R ** 2
            print('net.R', net.R)  # R^2=0.0081
            print('score', scores)
            if scores > 0: # threshold=0, can also compute other values using Fashion-MNIST w/ F-measure
                print('gen_img with name {0} is invalid'.format(img_path))
            else:
                print('gen_img with name {0} is valid'.format(img_path))
                #print('gen_img with name {0} is valid'.format(idx))
                valid_count += 1
                f.write('valid img {0}'.format(img_path))
                #f.write('valid img {0}'.format(idx))
                f.write('\n')
        f.write('gen_img path = {0}'.format(gen_img_folder))
        f.write('\n')
        f.write('total generated imgs: {0}, valid counts: {1}, % Valid: {2}'.format(len(img_paths), valid_count,
                                                                                    valid_count / len(img_paths)))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    ckpt_path = './deepSVDD_ckpt_soft_boundary_driving'
    gen_img_folder = './DLFuzz/Udacity/generated_inputs/Udacity'
    objective = 'soft-boundary'
    run_validation_deepSVDD(gen_img_folder, ckpt_path, objective)
