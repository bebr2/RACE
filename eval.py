import json
import sys

dataset_name = sys.argv[2]
model_name = sys.argv[1]

with open(f"./{model_name}_race_score.json") as f:
    data = json.load(f)
    
preds = [d["race_score"] for d in data]

with open(f"./modeloutput/{dataset_name}/{model_name}/judge.json") as f:
    y = json.load(f)
y = [yy["llm"] for yy in y]

import json
from sklearn.metrics import roc_auc_score


def get_AUC_ROC(preds, human_labels):
    auc_roc = roc_auc_score(human_labels, preds)
    return auc_roc * 100
print("AUC-ROC: ", get_AUC_ROC(preds, y))

