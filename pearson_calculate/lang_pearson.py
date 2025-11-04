import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import re
import numpy as np
from brain_mapping.roi_score import layer_roi_pearson_matrix, get_best_layer

# 默认bert-base-uncased
FMRI_ROOT = "filterData/lang/fmri"
EMB_DIR = "filterData/lang/design_matrix/deberta-base"
SAVE_ROOT = "results/lang/deberta-base"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 对齐时间
def align_fmri_embedding(fmri, emb):
    t_fmri = fmri.shape[0]
    t_emb = emb.shape[1] if emb.ndim == 3 else emb.shape[0]
    diff = t_fmri - t_emb

    if diff == 0:
        print(f"All good")
    elif diff > 0:
        fmri = fmri[:-diff, ...]
    else:
        emb = emb[:, :t_fmri, ...] if emb.ndim == 3 else emb[:t_fmri, ...]

    return fmri, emb



# 工具函数
def natural_sort_key(s):
    #按 run 编号自然排序
    match = re.search(r"run-(\d+)", s)
    return int(match.group(1)) if match else 9999


# 主流程
subs = sorted([s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-")])

group_results = []

for subj in subs:
    subj_dir = os.path.join(FMRI_ROOT, subj)
    subj_mats = []

    run_files = sorted(
        [f for f in os.listdir(subj_dir) if f.endswith("_bold_shared.npy")],
        key=natural_sort_key
    )

    if not run_files:
        print(f"No valid data")
        continue

    all_pearsons = []

    # section 按照顺序 1~9 匹配
    for idx, fmri_fname in enumerate(run_files[:9], 1):
        emb_path = os.path.join(EMB_DIR, f"lppEN_section{idx}_bold_embedding.npy")
        fmri_path = os.path.join(subj_dir, fmri_fname)

        if not os.path.exists(emb_path):
            continue

        print(f"🎯 Processing {subj} | run={fmri_fname} ↔ section{idx}")

        # 加载
        emb = np.load(emb_path, allow_pickle=True)
        fmri = np.load(fmri_path, allow_pickle=True)
        fmri, emb = align_fmri_embedding(fmri, emb)

        #计算 Pearson
        pearson_mat = layer_roi_pearson_matrix(emb, fmri, alpha=100.0, device="mps")
        all_pearsons.append(pearson_mat)

    if all_pearsons:
        subj_mean = np.mean(np.stack(all_pearsons, axis=0), axis=0)
        group_results.append(subj_mean)
    else:
        print(f"No valid data")

# 群体平均
if group_results:
    group_mean = np.mean(np.stack(group_results, axis=0), axis=0)
    np.save(os.path.join(SAVE_ROOT, "group_mean_layer_roi.npy"), group_mean)
    best_layer, roi_best_layers = get_best_layer(group_mean)
    np.save(os.path.join(SAVE_ROOT, "group_best_layer_roi.npy"), roi_best_layers)

else:
    print("No valid data")
