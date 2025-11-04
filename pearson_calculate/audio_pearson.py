import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import re
import numpy as np
from nilearn.glm.first_level import spm_hrf
from brain_mapping.roi_score import layer_roi_pearson_matrix, get_best_layer


#默认为wav2vec2-base-960h
FMRI_ROOT = "filterData/audio/fmri"
EMB_DIR = "filterData/audio/design_matrix/data2vec-audio-base"
SAVE_ROOT = "results/audio/data2vec-audio-base"
TR = 2.0
STANDARD_N_ROI = 178  

os.makedirs(SAVE_ROOT, exist_ok=True)

# 为音频embedding进行hrf卷积
def apply_hrf_and_resize(X, target_len, tr=2.0):
    hrf = spm_hrf(tr=tr)

    n_layer, n_time, dim = X.shape
    X_hrf = []

    for l in range(n_layer):
        X_conv = []
        for d in range(dim):
            x = np.convolve(X[l, :, d], hrf, mode='full')
            x = x[:target_len] if len(x) >= target_len else np.pad(x, (0, target_len - len(x)))
            X_conv.append(x)
        X_conv = np.stack(X_conv, axis=1)  # (target_len, dim)
        X_hrf.append(X_conv)

    return np.stack(X_hrf, axis=0)

# 对齐embedding与fmri
def align_and_prepare(emb, fmri, tr=2.0):
    emb = apply_hrf_and_resize(emb, fmri.shape[0], tr=tr)
    min_len = min(emb.shape[1], fmri.shape[0])
    return emb[:, :min_len, :], fmri[:min_len, :]

# 提取fmri所对应的任务名
def extract_task_name(filename):
    match = re.search(r"task-([a-zA-Z0-9]+)", filename)
    return match.group(1) if match else None


group_results = []

subs = sorted([s for s in os.listdir(FMRI_ROOT) if s.startswith("sub-")])

for subj in subs:
    subj_dir = os.path.join(FMRI_ROOT, subj)
    sessions = sorted([s for s in os.listdir(subj_dir) if s.startswith("ses-")])
    subj_mats = []
    print(subj,"Begin")

    for ses in sessions:
        ses_dir = os.path.join(subj_dir, ses)
        save_dir = os.path.join(SAVE_ROOT, subj, ses)

        fmri_files = [f for f in os.listdir(ses_dir) if f.endswith("_bold_shared_roi.npy")]
        all_pearsons = []

        for fname in fmri_files:
            task_name = extract_task_name(fname)
            if task_name is None:
                continue

            fmri_path = os.path.join(ses_dir, fname)
            emb_path = os.path.join(EMB_DIR, f"{task_name}.npy")

            if not os.path.exists(emb_path):
                continue

            #加载 fMRI
            fmri = np.load(fmri_path, allow_pickle=True)
            n_roi = fmri.shape[1]

            #ROI 数检查
            if n_roi != STANDARD_N_ROI:
                continue

            # 加载 embedding
            emb = np.load(emb_path, allow_pickle=True)
            emb, fmri = align_and_prepare(emb, fmri, tr=TR)

            # 计算 Pearson
            pearson_mat = layer_roi_pearson_matrix(emb, fmri, alpha=100.0, device="mps")
            all_pearsons.append(pearson_mat)

        # Session 平均
        if all_pearsons:
            mean_ses = np.mean(np.stack(all_pearsons, axis=0), axis=0)
            subj_mats.append(mean_ses)
        else:
            print(f"{subj}/{ses} skip")

    # Average all the participants
    if subj_mats:
        subj_mean = np.mean(np.stack(subj_mats, axis=0), axis=0)
        subj_save_dir = os.path.join(SAVE_ROOT, subj)
        group_results.append(subj_mean)
    else:
        print(f"{subj} Skip")


if group_results:
    group_mean = np.mean(np.stack(group_results, axis=0), axis=0)
    np.save(os.path.join(SAVE_ROOT, "group_mean_layer_roi.npy"), group_mean)
    best_layer, roi_best_layers = get_best_layer(group_mean)
    np.save(os.path.join(SAVE_ROOT, "group_roi_best_layers.npy"), roi_best_layers)
    print(f"group_mean shape: {group_mean.shape}")
else:
    print("没有任何有效结果（所有任务 ROI 数不匹配）。")
