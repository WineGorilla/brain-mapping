import os
import numpy as np
import torch
from tqdm import tqdm
from nilearn.glm.first_level import spm_hrf
from models.audio_encoder import load_audio, load_audio_model


def generate_audio_embeddings(
    model_name="facebook/wav2vec2-base-960h",
    stimuli_dir="data/audio_data/ds003020-download/stimuli",
    save_root="filterData/audio/design_matrix",
    tr=2.0,
    device="mps",
    sr_target=16000,
):

    # 初始化保存目录
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    print(f"Model Name: {model_name}")
    processor, model = load_audio_model(model_name, device)
    model.eval()

    # 提取函数
    def get_audio_embeddings(audio_path, processor, model, device, tr=2.0, sr_target=16000):
        """将一段音频按 TR 分块，提取多层 embedding 并返回 numpy 数组"""
        y, sr = load_audio(audio_path, target_sr=sr_target)
        chunk_size = int(sr * tr)
        chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

        n_layers = model.config.num_hidden_layers + 1
        layer_accum = [[] for _ in range(n_layers)]

        for chunk in tqdm(chunks, desc=f"🎧 {os.path.basename(audio_path)}"):
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            for l, h in enumerate(hidden_states):
                emb = h.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
                layer_accum[l].append(emb)

        # 拼接为 (n_layers, N_chunk, D)
        X_layers = np.stack([np.stack(layer_accum[l], axis=0) for l in range(n_layers)], axis=0)
        return X_layers

    # 遍历所有 .wav 文件
    wav_files = [f for f in os.listdir(stimuli_dir) if f.endswith(".wav")]

    for fname in wav_files:
        audio_path = os.path.join(stimuli_dir, fname)
        y, sr = load_audio(audio_path)

        # 提取 embedding
        X_layers = get_audio_embeddings(
            audio_path, processor, model, device=device,
            tr=tr, sr_target=sr_target
        )

        # 保存
        save_name = fname.replace(".wav", ".npy")
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, X_layers)

    print("\n Finish all, saved in:", save_dir)


# 调用
# 默认为facebook/wav2vec2-base-960h
if __name__ == "__main__":
    generate_audio_embeddings(
        model_name="facebook/data2vec-audio-base",
        device="mps",
        tr=2.0
    )
