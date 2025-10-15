import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def load_multimodal_model(model_name,device):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def get_multimodal_embeddings(processor, model, texts, image_paths, device, cls_only=True, batch_size=4):
    """
    批量提取跨模态（文本+图像）embedding
    返回:
        text_embeds: (T×D)
        image_embeds: (T×D)
        joint_embeds: (T×2D) 或 (T×D)
    """
    model.eval()
    text_embeds_gpu, image_embeds_gpu = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding multimodal pairs (batch)"):
        batch_texts = texts[i:i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]

        # CLIP processor 支持同时处理多文本多图像
        inputs = processor(text=batch_texts, images=batch_imgs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            t_emb = outputs.text_embeds.detach()   # (B, D)
            i_emb = outputs.image_embeds.detach()  # (B, D)

        text_embeds_gpu.append(t_emb)
        image_embeds_gpu.append(i_emb)

    # 拼接所有 batch，统一转CPU
    text_embeds = torch.cat(text_embeds_gpu, dim=0).cpu().numpy()
    image_embeds = torch.cat(image_embeds_gpu, dim=0).cpu().numpy()

    # 融合
    if cls_only:
        # 拼接方式保留模态信息 (T, 2D)
        joint_embeds = np.concatenate([text_embeds, image_embeds], axis=1)
    else:
        # 平均融合 (T, D)
        joint_embeds = (text_embeds + image_embeds) / 2.0

    return text_embeds, image_embeds, joint_embeds

