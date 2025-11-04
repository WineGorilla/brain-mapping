import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from brain_mapping.pls import *
import numpy as np

df = pd.read_csv("processed/BrainModelCovDataset.csv")
roi_cols = [c for c in df.columns if c.startswith("7Networks_")]
B = df[roi_cols].values  # shape = (n_models, n_ROI)

model_names = df["model_name"].tolist()
labels = df["modality"].tolist()

coords_pls, pls = run_pls_2d(B, labels)
plotp_pls_space_2d(coords_pls, model_names, labels,show_labels=False)
