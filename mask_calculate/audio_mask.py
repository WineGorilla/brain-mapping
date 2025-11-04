import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import nibabel as nib
from nilearn.image import math_img, resample_to_img

# 提取所有audio fmri数据每个文件的共享ROI

# 配置路径
ROOT = "data/audio_data/ds003020-download"
SAVE_ROOT = "filterData/shared_masks"
os.makedirs(SAVE_ROOT, exist_ok=True)


def make_subject_mask(sub_dir, save_path):
    nii_files = []
    for ses in sorted(os.listdir(sub_dir)):
        ses_func = os.path.join(sub_dir, ses, "func")
        if not os.path.exists(ses_func):
            continue
        for f in os.listdir(ses_func):
            if f.endswith(".nii.gz") and "bold" in f:
                nii_files.append(os.path.join(ses_func, f))

    if not nii_files:
        return None

    # 初始化参考mask
    try:
        ref_img = nib.load(nii_files[0])
        ref_mask = math_img("img > 0", img=ref_img.slicer[..., 0])
    except Exception as e:
        return None

    for i, fpath in enumerate(nii_files[1:], 2):
        try:
            img = nib.load(fpath)
            if img.affine is None or not (img.affine == ref_img.affine).all():
                img = resample_to_img(img, ref_img, interpolation="nearest",
                                      force_resample=True, copy_header=True)
            tmp_mask = math_img("img > 0", img=img.slicer[..., 0])
            ref_mask = math_img("a & b", a=ref_mask, b=tmp_mask)
        except Exception as e:
            print(f"跳过损坏文件: {fpath} ({e.__class__.__name__})")
    return ref_mask


# 遍历所有被试
subs = sorted([s for s in os.listdir(ROOT) if s.startswith("sub-")])

subject_masks = []
for sub in subs:
    sub_dir = os.path.join(ROOT, sub)
    save_path = os.path.join(SAVE_ROOT, f"{sub}_shared_mask.nii.gz")
    mask = make_subject_mask(sub_dir, save_path)
    if mask is not None:
        subject_masks.append(mask)


if len(subject_masks) > 1:
    group_mask = subject_masks[0]
    for m in subject_masks[1:]:
        try:
            m_res = resample_to_img(m, group_mask, interpolation="nearest",
                                    force_resample=True, copy_header=True)
            group_mask = math_img("a & b", a=group_mask, b=m_res)
        except Exception as e:
            print(f"跳过损坏被试掩膜 ({e.__class__.__name__})")

    group_path = os.path.join(SAVE_ROOT, "audio_group_shared_mask.nii.gz")
    group_mask.to_filename(group_path)
    print(f"群体共享掩膜已生成: {group_path}")
else:
    print("未生成足够的被试掩膜。")

