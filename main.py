import numpy as np
from datasets import load_dataset


from brain_mapping.cka_similarity import compute_model_similarity
from brain_mapping.dimensionality_reduction import reduce_model_space
from brain_mapping.visualize_space import plot_model_space,run_pca_2d,plot_model_space_origin_axes,plot_model_space


# 语言模型
# 语言模型
b_bert = np.load("results/lang/bert-base-uncased/group_best_layer_roi.npy")
roberta_base = np.load("results/lang/roberta-base/group_best_layer_roi.npy")
distilbert = np.load("results/lang/distilbert-base-uncased/group_best_layer_roi.npy")
albert_base_v2 = np.load("results/lang/albert-base-v2/group_best_layer_roi.npy")
roberta_large = np.load("results/lang/roberta-large/group_best_layer_roi.npy")
xlm_roberta_base = np.load("results/lang/xlm-roberta-base/group_best_layer_roi.npy")
bert_large_uncased = np.load("results/lang/bert-large-uncased/group_best_layer_roi.npy")
bert_base_multilingual_cased = np.load("results/lang/bert-base-multilingual-cased/group_best_layer_roi.npy")
xlm_roberta_large = np.load("results/lang/xlm-roberta-large/group_best_layer_roi.npy")
albert_large_v2 = np.load("results/lang/albert-large-v2/group_best_layer_roi.npy")
electra_base_discriminator = np.load("results/lang/electra-base-discriminator/group_best_layer_roi.npy")
electra_large_discriminator = np.load("results/lang/electra-large-discriminator/group_best_layer_roi.npy")
bert_base_cased = np.load("results/lang/bert-base-cased/group_best_layer_roi.npy")
bert_large_cased = np.load("results/lang/bert-large-cased/group_best_layer_roi.npy")
deberta_base = np.load("results/lang/deberta-base/group_best_layer_roi.npy")
deberta_large = np.load("results/lang/deberta-large/group_best_layer_roi.npy")


# 音频模型
wav2vec2 = np.load("results/audio/wav2vec2-base-960h/group_roi_best_layers.npy")
wavlm_base = np.load("results/audio/wavlm-base/group_roi_best_layers.npy")
hubert_base_ls960 = np.load("results/audio/hubert-base-ls960/group_roi_best_layers.npy")
wav2vec2_large_xlsr_53 = np.load("results/audio/wav2vec2-large-xlsr-53/group_roi_best_layers.npy")
data2vec_audio_base = np.load("results/audio/data2vec-audio-base/group_roi_best_layers.npy")
data2vec_audio_large = np.load("results/audio/data2vec-audio-large/group_roi_best_layers.npy")
hubert_large_ls960_ft = np.load("results/audio/hubert-large-ls960-ft/group_roi_best_layers.npy")
wavlm_base_plus = np.load("results/audio/wavlm-base-plus/group_roi_best_layers.npy")
wav2vec2_xls_r_1b = np.load("results/audio/wav2vec2-xls-r-1b/group_roi_best_layers.npy")
mms_300m = np.load("results/audio/mms-300m/group_roi_best_layers.npy")
wavlm_large = np.load("results/audio/wavlm-large/group_roi_best_layers.npy")
wav2vec2_base_superb_ks = np.load("results/audio/wav2vec2-base-superb-ks/group_roi_best_layers.npy")
wav2vec2_xls_r_300m = np.load("results/audio/wav2vec2-xls-r-300m/group_roi_best_layers.npy")

# 视觉模型
vit_base_patch14 = np.load("results/img/vit-base-patch14-224/group_roi_best_layers.npy")
vit_large_patch16 = np.load("results/img/vit-large-patch16-224/group_roi_best_layers.npy")
deit_base_patch16_224 = np.load("results/img/deit-base-patch16-224/group_roi_best_layers.npy")
dinov2_base = np.load("results/img/dinov2-base/group_roi_best_layers.npy")
data2vec_vision_base = np.load("results/img/data2vec-vision-base/group_roi_best_layers.npy")
dinov2_large = np.load("results/img/dinov2-large/group_roi_best_layers.npy")
dinov2_small = np.load("results/img/dinov2-small/group_roi_best_layers.npy")
beit_base_patch16_224 = np.load("results/img/beit-base-patch16-224/group_roi_best_layers.npy")
deit_small_patch16_224 = np.load("results/img/deit-small-patch16-224/group_roi_best_layers.npy")
beit_large_patch16_224 = np.load("results/img/beit-large-patch16-224/group_roi_best_layers.npy")
data2vec_vision_large = np.load("results/img/data2vec-vision-large/group_roi_best_layers.npy")
dino_vitb16 = np.load("results/img/dino-vitb16/group_roi_best_layers.npy")
dino_vits16 = np.load("results/img/dino-vits16/group_roi_best_layers.npy")
vit_mae_large = np.load("results/img/vit-mae-large/group_roi_best_layers.npy")
vit_mae_base = np.load("results/img/vit-mae-base/group_roi_best_layers.npy")
vit_msn_base = np.load("results/img/vit-msn-base/group_roi_best_layers.npy")
vit_msn_large = np.load("results/img/vit-msn-large/group_roi_best_layers.npy")

# 拼接为ROI矩阵
B = np.stack([b_bert,roberta_base,distilbert,albert_base_v2,roberta_large,xlm_roberta_base,bert_large_uncased,bert_base_multilingual_cased,xlm_roberta_large,albert_large_v2,electra_base_discriminator,electra_large_discriminator,bert_base_cased,bert_large_cased,deberta_base,deberta_large,
            wav2vec2,wavlm_base,hubert_base_ls960,wav2vec2_large_xlsr_53,data2vec_audio_base,data2vec_audio_large,hubert_large_ls960_ft,wavlm_base_plus,wav2vec2_xls_r_1b,mms_300m,wavlm_large,wav2vec2_base_superb_ks,wav2vec2_xls_r_300m,
            vit_base_patch14,vit_large_patch16,deit_base_patch16_224,dinov2_base,data2vec_vision_base,dinov2_large,dinov2_small,beit_base_patch16_224,deit_small_patch16_224,beit_large_patch16_224,data2vec_vision_large,dino_vitb16,dino_vits16,vit_mae_large,vit_mae_base,vit_msn_base,vit_msn_large])  # (N_ROI × 2)

model_names = ["BERT-base","roberta_base","distilbert","albert_base_v2","roberta_large","xlm_roberta_base","bert_large_uncased","bert_base_multilingual_cased","xlm_roberta_large","albert_large_v2","electra_base_discriminator","electra_large_discriminator","bert_base_cased","bert_large_cased","deberta_base","deberta_large",
               "wav2vec2-base","wavlm_base","hubert_base_ls960","wav2vec2_large_xlsr_53","data2vec_audio_base","data2vec_audio_large","hubert_large_ls960_ft","wavlm_base_plus","wav2vec2_xls_r_1b","mms_300m","wavlm_large","wav2vec2_base_superb_ks","wav2vec2_xls_r_300m",
               "vit-base-patch14","vit_large_patch16","deit_base_patch16_224","dinov2_base","data2vec_vision_base","dinov2_large","dinov2_small","beit_base_patch16_224","deit_small_patch16_224","beit_large_patch16_224","data2vec_vision_large","dino_vitb16","dino_vits16","vit_mae_large","vit_mae_base","vit_msn_base","vit_msn_large"]

# 定义模态标签
labels = (
    ["lang"] * 16 +      
    ["audio"] * 13 +     
    ["vision"] * 17    
)

#MDS
S = compute_model_similarity(B)
coords = reduce_model_space(S, method="MDS", n_components=3)
plot_model_space(coords, model_names, title="Model Representation in Brain Space (MDS)")


# PCA降维度
coords_3d, explained = run_pca_2d(B)
plot_model_space_origin_axes(coords_3d, model_names,labels=labels)
