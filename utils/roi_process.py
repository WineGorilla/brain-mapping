import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker

# 提取共享ROI
def extract_shared_roi_timeseries(
        fmri_path,
        atlas_path="atlas_schaefer_shared.nii.gz",
        label_names=None,
        modality_name="unknown",
        tr=2.0,
        save=True):

    # Load fMRI and shared ROI atlas
    fmri_img = nib.load(fmri_path)
    atlas_img = nib.load(atlas_path)

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,   # Z-score
        detrend=True,
        t_r=tr
    )

    # ROI signals extraction
    roi_ts = masker.fit_transform(fmri_img)  # (T, N_ROI)
    n_rois = roi_ts.shape[1]

    if label_names is None:
        label_names = [f"ROI_{i+1}" for i in range(n_rois)]
    else:
        label_names = label_names[:n_rois]

    if save:
        np.save(f"{modality_name}.npy", roi_ts)
        print(f"已保存: {modality_name}.npy")

    return roi_ts

