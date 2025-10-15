from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


def load_model(model_name,device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,output_hidden_states=True)
    model.to(device)
    model.eval()
    return tokenizer,model

def get_embeddings(tokenizer, model, texts, device, cls_only=True, all_layers=False, batch_size=8):
    """
    计算文本的 BERT embedding（支持多层与批量处理）
    参数：
        tokenizer, model: 预训练语言模型
        texts: list[str]
        device: "cpu"/"cuda"/"mps"
        cls_only: 是否取 [CLS] 向量
        all_layers: 是否返回所有层
        batch_size: 每批文本数量
    返回：
        如果 all_layers=True: list of (T×D)
        否则: (T×D)
    """
    model.config.output_hidden_states = True
    model.eval()

    all_hidden_layers = []   # 每批的所有层
    all_last_hidden = []     # 如果只取最后一层

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        if all_layers:
            hidden_states = outputs.hidden_states  # tuple(len=L+1)
            if cls_only:
                # 每层 (B, hidden)
                layer_embeds = [h[:, 0, :].detach() for h in hidden_states]
            else:
                # 每层 (B, seq_len, hidden)
                layer_embeds = [h.detach() for h in hidden_states]
            all_hidden_layers.append(layer_embeds)
        else:
            last_hidden = outputs.last_hidden_state.detach()  # (B, seq_len, hidden)
            all_last_hidden.append(last_hidden)

    if all_layers:
        n_layers = len(all_hidden_layers[0])
        # 拼接每层所有 batch
        layer_concat = [torch.cat([batch[i] for batch in all_hidden_layers], dim=0) for i in range(n_layers)]
        if cls_only:
            X_layers = [layer.cpu().numpy() for layer in layer_concat]  # list of (T×D)
        else:
            X_layers = [layer.cpu().numpy() for layer in layer_concat]  # list of (T×seq×D)
        return X_layers
    else:
        # 只取最后一层
        last_hidden_all = torch.cat(all_last_hidden, dim=0)  # (T, seq_len, D)
        if cls_only:
            X = last_hidden_all[:, 0, :].cpu().numpy()  # (T, D)
        else:
            X = last_hidden_all.cpu().numpy()           # (T, seq, D)
        return X

