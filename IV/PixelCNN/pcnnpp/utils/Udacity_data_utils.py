import time
from keras.preprocessing import image
import numpy as np
import random
from tqdm import tqdm
def preprocess(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    img_01 = input_img_data/255.        #input to VAE, shape (1,100,100,3)
    return img_01[0]


def load_train_data(path='。/datasets/Ch2_002/'):
    xs = []
    ys = []
    a = 0
    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + line.split(',')[5])
            ys.append(float(line.split(',')[6]))
            a+=1
            if a==101394:
               break
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_img_paths, gt_labels = np.array(xs), np.array(ys)
    train_data = [preprocess(img_path) for img_path in tqdm(train_img_paths)]
    return train_data, np.array(ys)



def load_test_data(path='。/Driving/testing/'):
    xs = []
    ys = []
    a = 0
    with open(path + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
            a+=1
            if a==5610:
            #if a == 60:
               break
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_img_paths, gt_labels = np.array(xs), np.array(ys)
    train_data = [preprocess(img_path) for img_path in tqdm(train_img_paths)]
    return train_data, np.array(ys)

