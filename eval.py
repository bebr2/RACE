import json
import os



with open("./race_scores.json") as f:
    data = json.load(f)
    
preds = [d["race_score"] for d in data]




dataset_name = "NQ"
model_name = "qwen7b"

with open(f"./modeloutput/{dataset_name}/{model_name}/judge.json") as f:
    y = json.load(f)
y = [yy["llm"] for yy in y]

import json
import os
import random
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def get_AUC_ROC(preds, human_labels):
    auc_roc = roc_auc_score(human_labels, preds)
    return auc_roc * 100
print("AUC-ROC: ", get_AUC_ROC(preds, y))

