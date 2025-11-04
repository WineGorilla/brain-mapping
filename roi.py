import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, math_img
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

# 提取共享

atlas_path = "filterData/shared_masks/atlas_schaefer_shared.nii.gz"

# 加载示例掩膜
mask_audio = nib.load("filterData/shared_masks/audio_group_shared_mask.nii.gz")
mask_image = nib.load("filterData/shared_masks/img_group_shared_mask.nii.gz")
mask_lang  = nib.load("filterData/shared_masks/lang_group_shared_mask.nii.gz")

# 对齐空间
mask_lang_res = resample_to_img(mask_lang, mask_audio, interpolation='nearest')
mask_image_res = resample_to_img(mask_image, mask_audio, interpolation='nearest')

# 生成共享体素掩膜
mask_shared = math_img("(a + b + c) >= 3", a=mask_audio, b=mask_image_res, c=mask_lang_res)
mask_shared.to_filename("shared_voxel_mask.nii.gz")

# 加载 Schaefer atlas ROI
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
atlas_img = nib.load(atlas.maps)
roi_names = atlas.labels
atlas_res = resample_to_img(atlas_img, mask_audio, interpolation='nearest')
atlas_shared = math_img("atlas * mask", atlas=atlas_res, mask=mask_shared)
atlas_shared.to_filename(atlas_path)
print("Shared ROI Templated has been created")

# 提取 ROI 信号函数
def extract_roi_signals(fmri_path, atlas_img, label_names, modality_name):
    fmri_img = nib.load(fmri_path)
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        mask_img="shared_voxel_mask.nii.gz",
        standardize=True,
        detrend=True,
        t_r=2.0
    )
    roi_ts = masker.fit_transform(fmri_img)

    if roi_ts.size == 0:
        return pd.DataFrame()

    valid_labels = label_names[1:roi_ts.shape[1]+1]
    df = pd.DataFrame(roi_ts, columns=valid_labels)
    df.to_csv(f"filterData/shared_masks/{modality_name}_roi_timeseries.csv", index=False)
    return df


# 提取示例文件的ROI
fmri_audio_path = "data/audio_data/ds003020-download/sub-UTS01/ses-3/func/sub-UTS01_ses-3_task-howtodraw_bold.nii.gz"
fmri_image_path = "data/image_data/ds004192-download/sub-01/ses-things03/func/sub-01_ses-things03_task-things_run-01_bold.nii.gz"
fmri_lang_path  = "data/language_data/sub-EN092_task-lppEN_run-10_space-MNIColin27_desc-preproc_bold.nii.gz"

df_audio = extract_roi_signals(fmri_audio_path, atlas_shared, roi_names, "audio")
df_image = extract_roi_signals(fmri_image_path, atlas_shared, roi_names, "image")
df_lang  = extract_roi_signals(fmri_lang_path,  atlas_shared, roi_names, "language")
