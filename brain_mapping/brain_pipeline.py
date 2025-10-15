import numpy as np
import torch
from models.language_encoder import load_model as load_text_model, get_embeddings as get_text_embeddings
from models.image_encoder import get_image_embeddings
from models.audio_encoder import get_audio_embeddings
from models.multimodal_encoder import get_multimodal_embeddings
from brain_mapping.encoding_model import evaluate_layers_cv, get_best_brain_score
from brain_mapping.roi_score import compute_model_fit_all_rois

def compute_brain_alignment(model_name, modality, inputs, y, roi_dict, device="cpu", alpha=1.0, n_splits=3, batch_size=8):
    """
    通用脑-模型对齐函数
    支持 text / image / audio / multimodal 模型
    
    参数：
        model_name : str
            模型名称，例如 "bert-base-uncased" / "facebook/wav2vec2-base" / "openai/clip-vit-base-patch16"
        modality : str
            模型类型，取值为 "text" / "image" / "audio" / "multimodal"
        inputs : dict
            模态输入：
              - text 模型: {"texts": [...]}  
              - image 模型: {"image_paths": [...]}  
              - audio 模型: {"audio_paths": [...]}  
              - multimodal 模型: {"texts": [...], "image_paths": [...]}  
        y : np.ndarray
            脑信号 (T×V)
        roi_dict : dict
            ROI 名称 → voxel 索引
        device : str
            "cpu" / "cuda" / "mps"
        alpha : float
            岭回归惩罚参数
        n_splits : int
            交叉验证折数
        batch_size : int
            批处理大小

    返回：
        best_layer, roi_vector, roi_scores
    """
    print(f"\n🧠 Processing {modality} model: {model_name}")

    # ===== Step 1: 加载模型并提取 embedding =====
    if modality == "text":
        tokenizer, model = load_text_model(model_name, device=device)
        X_layers = get_text_embeddings(tokenizer, model, inputs["texts"], device=device, all_layers=True, cls_only=True, batch_size=batch_size)

    elif modality == "image":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        X_layers = get_image_embeddings(processor, model, inputs["image_paths"], device=device, all_layers=True, cls_only=True)

    elif modality == "audio":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        X_layers = get_audio_embeddings(processor, model, inputs["audio_paths"], device=device, all_layers=True)

    elif modality == "multimodal":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        text_embeds, image_embeds, joint_embeds = get_multimodal_embeddings(processor, model, 
                                                                           inputs["texts"], inputs["image_paths"], device=device)
        X_layers = [joint_embeds]  # 只有一层融合 embedding

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # ===== Step 2: 计算 brain score =====
    scores = evaluate_layers_cv(X_layers, y, alpha=alpha, n_splits=n_splits, device=device)
    best_layer, best_score = get_best_brain_score(scores)
    print(f"✅ Best layer for {model_name}: {best_layer} (mean corr = {best_score:.4f})")

    # ===== Step 3: 取最佳层 embedding =====
    X_best = X_layers[best_layer]

    # ===== Step 4: ROI 拟合 =====
    roi_vector, roi_scores = compute_model_fit_all_rois(X_best, y, roi_dict, alpha=alpha)
    print(f"🏁 Finished {model_name}")

    for roi, score in roi_scores.items():
        print(f"   ROI {roi}: {score:.4f}")

    return best_layer, roi_vector, roi_scores

