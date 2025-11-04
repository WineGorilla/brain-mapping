import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nilearn.glm.first_level import spm_hrf
from models.image_encoder import load_image_model, get_image_embeddings


def generate_image_embeddings(
    model_name="google/vit-base-patch16-224",
    data_root="data/image_data/ds004192-download",
    img_root="data/image_data/images",
    save_root="filterData/img/design_matrix",
    tr=2.0,
    device="mps",
    batch_size=8,
):

    # 文件夹
    model_tag = model_name.split("/")[-1]
    model_save_root = os.path.join(save_root, model_tag)
    os.makedirs(model_save_root, exist_ok=True)

    extractor, model = load_image_model(model_name, device=device)

    subs = sorted(glob(os.path.join(data_root, "sub-*")))

    def process_run(events_file, sub, ses, run_tag):
        df = pd.read_csv(events_file, sep="\t")

        df = df[df["trial_type"].isin(["exp", "test"])].reset_index(drop=True)
        if len(df) == 0:
            return

        valid_rows, img_paths = [], []
        for _, row in df.iterrows():
            if isinstance(row.get("file_path", None), str):
                img_path = os.path.join(img_root, row["file_path"])
                if os.path.exists(img_path):
                    valid_rows.append(row)
                    img_paths.append(img_path)

        df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(df) == 0:
            return

        # 提取模型 embedding
        X_layers = get_image_embeddings(
            extractor, model, img_paths,
            device=device, all_layers=True, cls_only=True,
            batch_size=batch_size
        )
        n_layers = len(X_layers)
        feat_dim = X_layers[0].shape[1]

        # 获取 BOLD 信息
        bold_file = events_file.replace("_events.tsv", "_bold.nii.gz")
        if not os.path.exists(bold_file):
            return
        n_tr = nib.load(bold_file).shape[-1]

        # 对齐 & HRF 卷积
        df["tr_idx"] = (df["onset"] / tr).round().astype(int)
        hrf = spm_hrf(tr, oversampling=1)
        X_all = np.zeros((n_layers, n_tr, feat_dim))

        for li in range(n_layers):
            X_TR = np.zeros((n_tr, feat_dim))
            for si, row in df.iterrows():
                ti = row["tr_idx"]
                if 0 <= ti < n_tr:
                    X_TR[ti] = X_layers[li][si]

            # HRF 卷积
            for f in range(feat_dim):
                X_all[li, :, f] = np.convolve(X_TR[:, f], hrf, mode="full")[:n_tr]

        # 保存结果到结构化目录
        sub_save_dir = os.path.join(model_save_root, sub, ses)
        os.makedirs(sub_save_dir, exist_ok=True)

        bold_name = os.path.basename(events_file).replace("_events.tsv", "")
        save_name = bold_name + "_bold_embedding.npy"
        save_path = os.path.join(sub_save_dir, save_name)

        np.save(save_path, X_all)

    # 遍历所有 subjects/sessions/runs
    for sub_path in subs:
        sub = os.path.basename(sub_path)
        ses_list = sorted(glob(os.path.join(sub_path, "ses-things*")))

        for ses_path in ses_list:
            ses = os.path.basename(ses_path)
            func_dir = os.path.join(ses_path, "func")
            event_files = sorted(glob(os.path.join(func_dir, "*_events.tsv")))

            for ef in event_files:
                run_tag = [x for x in ef.split("_") if "run" in x][0]  # e.g. run-01
                process_run(ef, sub, ses, run_tag)

    print(f"\nAll Done, saved in: {model_save_root}")


# 调用
# 默认为google/vit-base-patch16-224
if __name__ == "__main__":
    generate_image_embeddings(
        model_name="openai/clip-vit-large-patch14",
        device="mps",
        batch_size=8
    )
