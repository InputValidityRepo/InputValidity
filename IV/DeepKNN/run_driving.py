# python3 run_driving.py
import mkl
import faiss
import os
import sys
print(sys.version)
print(sys.path)
print(sys.executable)
sys.path.append('./util')
from util.args_loader import get_args
from util import metrics
import torch

import numpy as np


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()
dim = 50
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

cache_name = './cache/Udacity/dim10/Driving_test_PilotNet_dim_10.npz'
id_npzfile = np.load(cache_name)
feat_log = id_npzfile['arr_0']

ood_feat_log_all = {}
out_datasets = ['DX_occl_SVDD', 'DX_light_SVDD', 'DX_blackout_SVDD', 'DLFuzz_SVDD']
for ood_dataset in out_datasets:
    cache_name = f"./cache_OOD_joint/{ood_dataset}_drivevsDriving_PilotNet_out_dim_10.npz"
    ood_npzfile = np.load(cache_name)
    ood_feat_log = ood_npzfile['arr_0']
    ood_feat_log_all[ood_dataset] = ood_feat_log
breakpoint()

normalizer = lambda x: x / (np.linalg.norm(x + 1e-10, ord=2, axis=-1, keepdims=True))

prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

ftest = prepos_feat(feat_log)

print('ftest', ftest.shape)
food_all = {}
for ood_dataset in out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

#################### KNN score OOD detection #################
index = faiss.IndexFlatL2(ftest.shape[1])
index.add(ftest)
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
        if ood_dataset == 'DX_light_SVDD':
            print('DX_light', ood_dataset)
            gen_img_folder = './DX_SVDD_driving0/light/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_blackout_SVDD':
            print('DX_blackout', ood_dataset)
            gen_img_folder = './DX_SVDD_driving0/blackout/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DX_occl_SVDD':
            print('DX_occl', ood_dataset)
            gen_img_folder = './DX_SVDD_driving0/occl/0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        elif ood_dataset == 'DLFuzz_SVDD':
            print('DLFuzz', ood_dataset)
            gen_img_folder = './DLFuzz/Udacity/test_inputs_DLFuzz_SVDD/Udacity_0.1'
            img_paths = [img for img in os.listdir(gen_img_folder) if img.endswith(".png")]
            img_paths.sort()
        print('num of imgs', len(img_paths))
        results = metrics.cal_metric(scores_in, scores_ood_test, img_paths, None, ood_dataset, K)
        all_results.append(results)

    metrics.print_all_results(all_results, out_datasets, f'KNN k={K}')



