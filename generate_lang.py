import os
import numpy as np
import pandas as pd
import torch
from nilearn.glm.first_level import spm_hrf
from transformers import AutoTokenizer, AutoModel


def generate_language_embeddings(
    csv_path="data/language_data/EN/lppEN_word_information.csv",
    save_root="filterData/lang/design_matrix",
    model_name="bert-base-uncased",
    tr=2.0,
    device=None,
    batch_size=16,
):

    # 设备与模型准备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model_tag = model_name.split("/")[-1]
    save_dir = os.path.join(save_root, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    required_cols = {"onset", "offset", "word", "section"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{required_cols - set(df.columns)}")

    df = df.sort_values(["section", "onset"]).reset_index(drop=True)
    sections = sorted(df["section"].unique())

    # 提取 BERT embedding
    def get_text_embeddings(words):
        """提取每个词的多层 BERT embedding, 取 [CLS])"""
        all_layers = []
        with torch.no_grad():
            for i in range(0, len(words), batch_size):
                batch = words[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=32).to(device)
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states  # tuple of (layer, batch, seq, dim)
                cls_layers = [h[:, 0, :].cpu().numpy() for h in hidden_states]
                if not all_layers:
                    all_layers = [x for x in cls_layers]
                else:
                    for li in range(len(cls_layers)):
                        all_layers[li] = np.concatenate([all_layers[li], cls_layers[li]], axis=0)
        return np.stack(all_layers, axis=0)  # (n_layers, n_words, dim)

    # 遍历每个 section
    for sec in sections:
        sub_df = df[df["section"] == sec].reset_index(drop=True)
        words = sub_df["word"].astype(str).tolist()
        n_words = len(words)

        X_layers = get_text_embeddings(words)
        n_layers, n_words, feat_dim = X_layers.shape

        max_time = sub_df["offset"].max()
        n_tr = int(np.ceil(max_time / tr))
        sub_df["tr_idx"] = (sub_df["onset"] / tr).round().astype(int)

        X_TR = np.zeros((n_layers, n_tr, feat_dim))
        for li in range(n_layers):
            for si, row in sub_df.iterrows():
                ti = int(row["tr_idx"])
                if 0 <= ti < n_tr:
                    X_TR[li, ti, :] += X_layers[li, si, :]

        # HRF 卷积
        hrf = spm_hrf(tr, oversampling=1)
        X_hrf = np.zeros_like(X_TR)
        for li in range(n_layers):
            for f in range(feat_dim):
                X_hrf[li, :, f] = np.convolve(X_TR[li, :, f], hrf, mode="full")[:n_tr]

        save_path = os.path.join(save_dir, f"lppEN_section{sec}_bold_embedding.npy")
        np.save(save_path, X_hrf)

    print(f"\n All Done, saved in: {save_dir}")


# 调用
if __name__ == "__main__":
    generate_language_embeddings(
        model_name="microsoft/deberta-large",
        tr=2.0,
        device="mps",
        batch_size=16
    )
