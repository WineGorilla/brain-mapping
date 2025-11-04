import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from glob import glob
from utils.roi_process import extract_shared_roi_timeseries

# 配置路径
root_dir = "data/language_data/derivatives"
save_root = "filterData/lang/fmri"
atlas_path = "filterData/shared_masks/atlas_schaefer_shared.nii.gz"
os.makedirs(save_root, exist_ok=True)

subs = sorted(glob(os.path.join(root_dir, "sub-EN*")))
print("Subjects found:", subs)

for sub in subs:
    sub_name = os.path.basename(sub)
    func_dir = os.path.join(sub, "func")

    bold_files = sorted(glob(os.path.join(
        func_dir, "*task-lppEN*_desc-preproc_bold.nii.gz")))

    if not bold_files:
        continue

    out_dir = os.path.join(save_root, sub_name)
    os.makedirs(out_dir, exist_ok=True)

    for bold in bold_files:
        fname = os.path.basename(bold).replace(".nii.gz", "")
        modality_path = os.path.join(out_dir, f"{fname}_shared")

        extract_shared_roi_timeseries(
            fmri_path=bold,
            atlas_path=atlas_path,
            modality_name=modality_path,
            tr=2.0,
            save=True
        )

print("\n语言模态提取完成！保存在 filterData/lang/fmri/")

