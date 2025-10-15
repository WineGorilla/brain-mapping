import torch
from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_image_model(model_name,deivce):
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,output_hidden_states=True)
    model.to(deivce)
    model.eval()
    return extractor, model

def get_image_embeddings(extractor, model, image_paths, device, all_layers=True, cls_only=True, batch_size=4):
    """
    计算输入图片的多层embedding
    输出：
        X_layers: list of (T×D) numpy 数组，每层一个矩阵
    """
    all_hidden_layers = []
    model.eval()

    # 分批处理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = extractor(images=batch_imgs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple (L+1), 每层 (B, seq, D)

        if cls_only:
            # 取 [CLS] token → (B, hidden)
            layer_embeds = [h[:, 0, :].detach() for h in hidden_states]
        else:
            # 平均所有 patch → (B, hidden)
            layer_embeds = [h.mean(dim=1).detach() for h in hidden_states]

        # 加入结果（每个batch的层结构）
        all_hidden_layers.append(layer_embeds)

    # 重新组织结构：layer-first
    n_layers = len(all_hidden_layers[0])  # 每个样本层数相同
    # 拼接每层的所有 batch
    layer_concat = [torch.cat([batch[i] for batch in all_hidden_layers], dim=0) for i in range(n_layers)]

    # 转 numpy
    X_layers = [layer.cpu().numpy() for layer in layer_concat]

    # 输出与语言模型格式一致
    return X_layers


