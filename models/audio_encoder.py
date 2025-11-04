import torch
import numpy as np
import torchaudio
from transformers import AutoProcessor, AutoFeatureExtractor, AutoModel, Wav2Vec2Processor
from tqdm import tqdm
import librosa

def load_audio_model(model_name="facebook/wav2vec2-base-960h", device="mps"):
    print(f"加载音频模型: {model_name}")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    return processor, model

def load_audio(file_path, target_sr=16000):
    """读取音频文件并重采样为目标采样率"""
    waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if waveform_np is None or len(waveform_np) == 0:
        raise ValueError(f"Empty waveform at {file_path}")
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    return waveform, sr


# 计算embedding
def get_audio_embeddings(processor, model, audio_paths, device, all_layers=True, mean_pool=True, batch_size=4):
    all_hidden_layers = []
    model.eval()

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding audio (batch mode)"):
        batch_paths = audio_paths[i:i + batch_size]
        waveforms = []

        for path in batch_paths:
            w, sr = load_audio(path, target_sr=16000)
            w = w.squeeze()
            if w.ndim > 1:
                w = w.mean(dim=0)
            waveforms.append(w.cpu().numpy())  

        inputs = processor(
            waveforms,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple(L+1) of (B, T, D)

        # 平均池化每层 (B, D)
        if mean_pool:
            layer_embeds = [h.mean(dim=1).detach() for h in hidden_states]
        else:
            layer_embeds = [h.detach() for h in hidden_states]  # (B, T, D)

        all_hidden_layers.append(layer_embeds)

    # 合并每一层的所有 batch
    if not all_hidden_layers:
        raise ValueError("No embeddings extracted — check your audio paths or sampling.")

    n_layers = len(all_hidden_layers[0])
    layer_concat = [
        torch.cat([batch[i] for batch in all_hidden_layers], dim=0)
        for i in range(n_layers)
    ]

    X_layers = [layer.cpu().numpy() for layer in layer_concat]
    return X_layers



def get_audio_embeddings(audio_path, processor, model, device,
                         chunk_dur=20, sr_target=16000):
    model.eval()

    #加载音频
    y, sr = load_audio(audio_path, target_sr=sr_target)
    chunk_size = int(chunk_dur * sr)
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    n_chunks = len(chunks)

    #初始化每层容器
    n_layers = model.config.num_hidden_layers + 1
    layer_accum = [[] for _ in range(n_layers)]

    #分块处理
    for c, chunk in enumerate(tqdm(chunks, desc="Encoding chunks")):
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple(L+1), 每层 (1, T, D)

        # mean pooling 得到 (D,)
        for l, h in enumerate(hidden_states):
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
            layer_accum[l].append(emb)

    # 拼接所有chunk，(n_layers, N_chunk, D)
    X_layers = np.stack([np.stack(layer_accum[l], axis=0) for l in range(n_layers)], axis=0)

    print(f"输出形状: {X_layers.shape} = (n_layers, N_chunk, D)")
    return X_layers
