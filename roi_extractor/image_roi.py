import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from glob import glob
from utils.roi_process import extract_shared_roi_timeseries

# 提取视觉任务ROI
root_dir = "data/image_data/ds004192-download"
save_root = "filterData/img/fmri"
atlas_path = "filterData/shared_masks/atlas_schaefer_shared.nii.gz"
os.makedirs(save_root, exist_ok=True)

subs = sorted(glob(os.path.join(root_dir, "sub-*")))

for sub in subs:
    sub_name = os.path.basename(sub)

    ses_dirs = sorted(glob(os.path.join(sub, "ses-things*")))

    for ses in ses_dirs:
        ses_name = os.path.basename(ses)
        func_dir = os.path.join(ses, "func")

        bold_files = sorted(glob(os.path.join(func_dir, "*task-things*_bold.nii.gz")))
        if not bold_files:
            continue

        # 创建输出目录
        out_dir = os.path.join(save_root, sub_name, ses_name)
        os.makedirs(out_dir, exist_ok=True)

        for bold in bold_files:
            fname = os.path.basename(bold).replace(".nii.gz", "")
            modality_path = os.path.join(out_dir, f"{fname}_shared")


            extract_shared_roi_timeseries(
                fmri_path=bold,
                atlas_path=atlas_path,  # 选ROI mask
                modality_name=modality_path,
                tr=2.0,
                save=True
            )

print("\n全部被试提取完成！结果保存在 filterData/img/fmri/")

