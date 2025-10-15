import numpy as np
import torch
from datasets import load_dataset
import soundfile as sf
import os,re

from brain_mapping.cka_similarity import compute_model_similarity
from brain_mapping.dimensionality_reduction import reduce_model_space
from brain_mapping.visualize_space import plot_model_space
from brain_mapping.brain_pipeline import compute_brain_alignment

# 假设有10个时间点，voxel的数量为200个
n_samples = 10
n_voxels = 200
y = np.random.randn(n_samples, n_voxels)  # 模拟脑信号
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available() and torch.version.cuda:
    device = "cuda"
else:
    device = "cpu"

# 模拟文本数据
texts = [
    "The cat sat on the mat.",
    "A dog barked loudly.",
    "Birds are flying in the sky.",
    "Children are playing in the park.",
    "The sun rises in the east.",
    "The man opened the door.",
    "She drank a cup of coffee.",
    "The car stopped at the light.",
    "People are walking downtown.",
    "He is reading a newspaper."
]

# 模拟文本所对应的voxel
# 200个voxel，平均划分成20个ROI
roi_dict = {
    f"ROI_{i+1}": np.arange(i * 10, (i + 1) * 10)
    for i in range(20)
}

# 两个语言模型
_, b_bert, _ = compute_brain_alignment(
    model_name="bert-base-uncased",
    modality="text",
    inputs={"texts": texts},
    y=y,
    roi_dict=roi_dict,
    device="mps"
)

_, b_roberta, _ = compute_brain_alignment(
    model_name="roberta-base",
    modality="text",
    inputs={"texts": texts},
    y=y,
    roi_dict=roi_dict,
    device="mps"
)

# 视觉模型
dataset = load_dataset("cifar10", split="test[:20]")  # 取前20张
image_paths = []
texts = []

for i, item in enumerate(dataset):
    img = item["img"]
    label = dataset.features["label"].int2str(item["label"])
    file_name = f"image_{i}_{label}.jpg"
    save_path = os.path.join("images", file_name)
    img.save(save_path)
    image_paths.append(save_path)
    texts.append(f"{label}")
T = len(image_paths)   # 时间点 = 图片数
V = 200                # voxel 数量
y = np.random.randn(T, V)  # 随机脑信号矩阵
roi_image_dict = {
    f"ROI_{i+1}": np.arange(i * 10, (i + 1) * 10)
    for i in range(20)
}

best_layer_clip, roi_vector_clip, roi_scores_clip = compute_brain_alignment(
    model_name="google/vit-base-patch16-224",
    modality="image",
    inputs={"image_paths": image_paths},
    y=y,
    roi_dict=roi_image_dict,
    device="mps" 
)


# 音频模型
sr = 16000
audio_paths, texts = [], []

for i in range(10):
    waveform = np.random.randn(sr) * 0.01  # 1 秒噪音
    path = f"audios/test_{i}.wav"
    sf.write(path, waveform, sr)
    audio_paths.append(path)
    texts.append(f"random_{i}")

T = len(audio_paths)   # 时间点 = 音频条目数
V = 200                # voxel 数量
y = np.random.randn(T, V)  # 随机脑信号矩阵

roi_audio_dict = {
    f"ROI_{i+1}": np.arange(i * 10, (i + 1) * 10)
    for i in range(20)
}

best_layer_wavlm, roi_vector_wavlm, roi_scores_wavlm = compute_brain_alignment(
    model_name="facebook/wav2vec2-base-960h",
    modality="audio",
    inputs={"audio_paths": audio_paths},
    y=y,
    roi_dict=roi_audio_dict,
    device="mps"
)



# 多模态模型
image_dir = "./images"
image_paths = []
image_texts = []

for fname in sorted(os.listdir(image_dir)):
    if fname.endswith(".jpg"):
        # 完整路径
        path = os.path.join(image_dir, fname)
        image_paths.append(path)
        
        # 从文件名中提取类别词（_ 和 . 之间的那一段）
        match = re.search(r"_(.*?)\.jpg$", fname)
        if match:
            label = match.group(1)
        else:
            label = "unknown"
        
        # 构造自然语言描述，让 CLIP 更容易理解
        image_texts.append(f"a photo of a {label}")

T = len(image_paths)   # 时间点 = 音频条目数
V = 200                # voxel 数量
y = np.random.randn(T, V)  # 随机脑信号矩阵

best_layer_clip, roiclip, roi_scores_clip = compute_brain_alignment(
    model_name="openai/clip-vit-base-patch16",
    modality="multimodal",
    inputs={"texts": image_texts, "image_paths": image_paths},
    y=y,
    roi_dict=roi_dict,
    device="mps"
)

# 拼接为ROI矩阵
B = np.stack([b_bert, b_roberta,roi_vector_clip,roi_vector_wavlm,roiclip], axis=1)  # (N_ROI × 2)

S = compute_model_similarity(B)
coords = reduce_model_space(S, method="MDS", n_components=3)

print(coords)

model_names = ["BERT-base", "RoBERTa-base","vit-base-patch16-224","wavlm-base","clip-vit-base-patch16"]

# 绘制3D空间分布
plot_model_space(coords, model_names, title="Model Representation in Brain Space (MDS)")