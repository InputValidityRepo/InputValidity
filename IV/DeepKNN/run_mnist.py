# python run_mnist.py --in-dataset MNIST --name sinvad_CNN  --model-arch sinvad_CNN
import mkl
import faiss
import os
import time
from util.args_loader import get_args
from util import metrics
import torch

import numpy as np
import sys
print(sys.version)
print(sys.path)
print(sys.executable)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

cache_name = "./cache/MNIST/MNIST_train_sinvad_CNN_in_dim_512_alllayers.npz"
id_npzfile = np.load(cache_name)
feat_log, score_log, label_log = id_npzfile['arr_0'], id_npzfile['arr_1'], id_npzfile['arr_2']
class_num = score_log.shape[1]

cache_name = "./cache/MNIST/MNIST_val_sinvad_CNN_in_dim_512_alllayers.npz"
id_npzfile_val = np.load(cache_name)
feat_log_val, score_log_val, label_log_val = id_npzfile_val['arr_0'], id_npzfile_val['arr_1'], id_npzfile_val['arr_2']

ood_feat_log_all = {}
#out_datasets = ['DX_occl', 'DX_light', 'DX_blackout', 'SINVAD']
out_datasets = ['DX_black_DAIV']
for ood_dataset in out_datasets:
    cache_name = f"./cache/MNIST/{ood_dataset}vsMNIST_sinvad_CNN_out_dim_512_alllayers.npz"
    ood_npzfile = np.load(cache_name)
    ood_feat_log, ood_score_log = ood_npzfile['arr_0'], ood_npzfile['arr_1']
    ood_feat_log_all[ood_dataset] = ood_feat_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, 192:704]))

ftrain = prepos_feat(feat_log)   # in_dataset train feat
ftest = prepos_feat(feat_log_val)  # in_dataset val feat
food_all = {}
for ood_dataset in out_datasets:
    ood_feature = prepos_feat(ood_feat_log_all[ood_dataset])
    food_all[ood_dataset] = ood_feature
    print(ood_dataset, ood_feature.shape)
print('ftrain, ftest', ftrain.shape, ftest.shape)
time.sleep(10)

#################### KNN score OOD detection #################
#import faiss
index = faiss.IndexFlatL2(ftrain.shape[1])
index.add(ftrain)
for K in [1, 10, 20, 50, 100, 200, 500, 1000, 3000, 5000]:
    print('K', K)
    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    all_results = []
    all_score_ood = []
    for ood_dataset, food in food_all.items():
        D, _ = index.search(food, K)
        scores_ood_test = -D[:,-1]
        all_score_ood.extend(scores_ood_test)
        if ood_dataset == 'SINVAD':
            print('SINVAD', ood_dataset)
            img_paths = np.arange(100)
        elif ood_dataset == 'DX_light':
            print('DX_light', ood_dataset)
            gen_img_folder = '/light_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_blackout':
            print('DX_blackout', ood_dataset)
            gen_img_folder = '/blackout_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_black_DAIV':
            print('DX_blackout', ood_dataset)
            gen_img_folder = '/MNIST/DX_DAIV0/blackout/-2708.34_0.01'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_occl':
            print('DX_occl', ood_dataset)
            gen_img_folder = '/occl_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DLFuzz':
            print('DLFuzz', ood_dataset)
            gen_img_folder = '/DLFuzz_test_suite'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_light_SVDD':
            gen_img_folder = '/MNIST/DX_SVDD0/light/0_0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        results = metrics.cal_metric(scores_in, scores_ood_test, img_paths, None, ood_dataset, K)
        all_results.append(results)

    metrics.print_all_results(all_results, out_datasets, f'KNN k={K}')
    print()
