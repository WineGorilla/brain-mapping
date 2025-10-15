import torch
import numpy as np
import torchaudio
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import librosa

def load_audio_model(model_name,device):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    return processor, model

def load_audio(file_path, target_sr=16000):
    """读取音频文件并重采样为目标采样率"""
    waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if waveform_np is None or len(waveform_np) == 0:
        raise ValueError(f"Empty waveform at {file_path}")
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    return waveform, sr



def get_audio_embeddings(processor, model, audio_paths, device, all_layers=True, mean_pool=True, batch_size=4):
    """
    批量计算音频的多层 embedding。
    """
    all_hidden_layers = []
    model.eval()

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding audio (batch mode)"):
        batch_paths = audio_paths[i:i + batch_size]
        waveforms = []

        # ✅ 读取并保证每个音频为一维数组
        for path in batch_paths:
            w, sr = load_audio(path, target_sr=16000)
            w = w.squeeze()
            if w.ndim > 1:
                w = w.mean(dim=0)
            waveforms.append(w.cpu().numpy())  # 转 numpy 保证 processor 能识别

        # ✅ 确保 processor 接收 list[np.ndarray]，形状如 [(16000,), (18000,), ...]
        inputs = processor(
            waveforms,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(device)

        # ✅ 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple(L+1) of (B, T, D)

        # ✅ 平均池化每层 (B, D)
        if mean_pool:
            layer_embeds = [h.mean(dim=1).detach() for h in hidden_states]
        else:
            layer_embeds = [h.detach() for h in hidden_states]  # (B, T, D)

        all_hidden_layers.append(layer_embeds)

    # ✅ 合并每一层的所有 batch
    if not all_hidden_layers:
        raise ValueError("No embeddings extracted — check your audio paths or sampling.")

    n_layers = len(all_hidden_layers[0])
    layer_concat = [
        torch.cat([batch[i] for batch in all_hidden_layers], dim=0)
        for i in range(n_layers)
    ]

    # ✅ 转 CPU + NumPy
    X_layers = [layer.cpu().numpy() for layer in layer_concat]
    return X_layers
