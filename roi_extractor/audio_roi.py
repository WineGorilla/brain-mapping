import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from glob import glob
from utils.roi_process import extract_shared_roi_timeseries


# 基本路径配置
root_dir = "data/audio_data/ds003020-download"
save_root = "filterData/audio/fmri"
atlas_path = "filterData/shared_masks/atlas_schaefer_shared.nii.gz"
os.makedirs(save_root, exist_ok=True)

# 获取所有被试目录
subs = sorted(glob(os.path.join(root_dir, "sub-*")))

# 主循环
for sub in subs:
    sub_name = os.path.basename(sub)
    ses_dirs = sorted(glob(os.path.join(sub, "ses-*")))

    # 跳过 ses-1
    ses_dirs = [s for s in ses_dirs if not s.endswith("ses-1")]

    for ses in ses_dirs:
        ses_name = os.path.basename(ses)
        func_dir = os.path.join(ses, "func")

        if not os.path.exists(func_dir):
            continue

        # 获取所有 BOLD 文件
        bold_files = sorted(glob(os.path.join(func_dir, "*_task-*_bold.nii.gz")))
        if not bold_files:
            print(f"{sub_name}/{ses_name} No valid data")
            continue

        # 创建输出目录
        out_dir = os.path.join(save_root, sub_name, ses_name)
        os.makedirs(out_dir, exist_ok=True)

        for bold in bold_files:
            fname = os.path.basename(bold).replace(".nii.gz", "")
            modality_path = os.path.join(out_dir, f"{fname}_shared_roi")

            try:
                extract_shared_roi_timeseries(
                    fmri_path=bold,
                    atlas_path=atlas_path,  # 共享ROI模板
                    modality_name=modality_path,
                    tr=2.0,
                    save=True
                )
            except EOFError:
                continue
            except Exception as e:
                continue

print("\n全部被试的 Audio ROI 提取完成！结果保存在 filterData/audio/fmri/")
