import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import re
import numpy as np
from brain_mapping.roi_score import layer_roi_pearson_matrix, get_best_layer

# 配置路径
FMRI_ROOT = "filterData/img/fmri"
EMB_ROOT = "filterData/img/design_matrix/vit-msn-large"
SAVE_ROOT = "results/img/vit-msn-large"
STANDARD_N_ROI = 178
os.makedirs(SAVE_ROOT, exist_ok=True)


# 遍历所有
pearson_list = []

subs = sorted([s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-")])

for subj in subs:
    subj_dir = os.path.join(FMRI_ROOT, subj)
    for ses in sorted([s for s in os.listdir(subj_dir) if s.startswith("ses-")]):
        ses_dir = os.path.join(subj_dir, ses)
        fmri_files = [f for f in os.listdir(ses_dir) if f.endswith("_bold_shared.npy")]

        for fname in fmri_files:
            run_tag = re.search(r"run-\d+", fname)
            if not run_tag:
                continue

            fmri_path = os.path.join(ses_dir, fname)
            emb_path = os.path.join(EMB_ROOT, subj, ses,
                                    fname.replace("_bold_shared.npy", "_bold_embedding.npy"))

            if not os.path.exists(emb_path):
                continue

            fmri = np.load(fmri_path, allow_pickle=True)
            emb = np.load(emb_path, allow_pickle=True)

            if fmri.shape[1] != STANDARD_N_ROI:
                continue

            pearson = layer_roi_pearson_matrix(emb, fmri, alpha=1000.0, device="mps")
            pearson_list.append(pearson)

# 聚合结果并保存
if pearson_list:
    group_mean = np.mean(np.stack(pearson_list, axis=0), axis=0)
    np.save(os.path.join(SAVE_ROOT, "group_mean_layer_roi.npy"), group_mean)

    best_layer, roi_best_layers = get_best_layer(group_mean)
    np.save(os.path.join(SAVE_ROOT, "group_roi_best_layers.npy"), roi_best_layers)

else:
    print("No valid data")
