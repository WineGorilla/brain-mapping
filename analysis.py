import os
import numpy as np
import pandas as pd

# 读取 ROI 标签
roi_labels = list(pd.read_csv("filterData/shared_masks/audio_roi_timeseries.csv").columns)

# 路径
root_dir = "results"
modalities = {
    "lang": os.path.join(root_dir, "lang"),
    "audio": os.path.join(root_dir, "audio"),
    "vision": os.path.join(root_dir, "img"),
}

records = []

for modality, path_root in modalities.items():

    # 遍历子文件夹
    for model_name in sorted(os.listdir(path_root)):
        model_dir = os.path.join(path_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 自动判断文件名
        candidates = [
            os.path.join(model_dir, "group_roi_best_layers.npy"),
            os.path.join(model_dir, "group_best_layer_roi.npy"),
        ]
        best_path = next((p for p in candidates if os.path.exists(p)), None)

        if best_path is None:
            continue

        # 加载 ROI 数据
        arr = np.load(best_path)
        arr = np.squeeze(arr)

        if len(arr) != len(roi_labels):
            continue

        size = "base" if "base" in model_name else ("large" if "large" in model_name else "other")
        family = model_name.split("-")[0].lower()

        rec = {
            "model_name": model_name,
            "modality": modality,
            "family": family,
            "size": size,
        }
        rec.update({roi: val for roi, val in zip(roi_labels, arr)})
        records.append(rec)

# 转为 DataFrame
df = pd.DataFrame(records)

# 保存结果
os.makedirs("processed", exist_ok=True)
df.to_parquet("processed/BrainModelCovDataset.parquet", index=False)
df.to_csv("processed/BrainModelCovDataset.csv", index=False)


print(df.head(3))
