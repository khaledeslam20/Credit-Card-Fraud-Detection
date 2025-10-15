from sklearn.metrics import (f1_score, precision_recall_curve, auc,confusion_matrix, classification_report, make_scorer)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os
import pandas as pd
def evaluate_model(model, name ,x_train, y_train, x_val, y_val) :
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]

    f1_train = f1_score(y_train, y_train_pred)

    precision_train, recall_train ,threshold = precision_recall_curve(y_train, y_train_proba)
    pr_auc_train = auc(recall_train, precision_train) # pr_auc_train = auc(*precision_recall_curve(y_train, y_train_prob)[:2])

    y_val_pred = model.predict(x_val)
    y_val_proba = model.predict_proba(x_val)[:, 1]

    f1_val = f1_score(y_val, y_val_pred)
    precision_val, recall_val ,thresholds = precision_recall_curve(y_val, y_val_proba)

    pr_auc_val = auc(recall_val, precision_val)

    print(f"\n{name} Results:")
    print(f"Train F1 Score: {f1_train:.4f}, PR AUC: {pr_auc_train:.4f}")
    print(f"Val   F1 Score: {f1_val:.4f}, PR AUC: {pr_auc_val:.4f}")

    # print("\nTraining Classification Report:")
    # print(classification_report(y_train, y_train_pred, digits=4))
    #
    # print("\nValidation Classification Report:")
    # print(classification_report(y_val, y_val_pred, digits=4))
    return f1_val, pr_auc_val, thresholds, y_val_proba, y_val_pred,y_train_pred,y_train_proba,f1_train


def find_best_threshold (y_true, y_probs) :
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def save_model(model, scaler, threshold, model_name, path='model.pkl', config=None):
    model_dict = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "model_name": model_name,
        "config": config
    }
    with open(path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Model saved to {path}")






#
# def save_model(model, scaler, threshold, model_name, path='model.pkl', config=None):
#     model_dict = {
#         "model": model,
#         "scaler": scaler,
#         "threshold": threshold,
#         "model_name": model_name,
#         "config": config
#     }
#     with open(path, 'wb') as f:
#         pickle.dump(model_dict, f)
#     print(f"Model saved to {path}")

def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        model_dict = pickle.load(f)
    print(f"Model loaded from {path}")
    return model_dict


def predict_with_loaded_model(model_dict, x):
    scaler = model_dict.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)
    model = model_dict["model"]
    threshold = model_dict.get("threshold", 0.5)
    y_probs = model.predict_proba(x)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    return y_pred, y_probs


def save_confusion_matrix(y_true, y_pred, model_name, experiment_name, save_dir=".",suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    suffix_part = f"_{suffix}" if suffix else ""

    plt.title(f"Confusion Matrix - {model_name} ({experiment_name}{suffix_part})")
    filename = f"{save_dir}/conf_matrix_{experiment_name}_{model_name}{suffix_part}.png"
    plt.savefig(filename)
    plt.close()
    # print(f"Confusion matrix saved to {filename}")
    # plt.title(f"Confusion Matrix - {model_name} ({experiment_name})")
    # filename = f"{save_dir}/conf_matrix_{experiment_name}_{model_name}.png"
    # plt.savefig(filename)
    # plt.close()
    # print(f"Confusion matrix saved to {filename}")

def plot_f1_heatmap(experiment_names, results_dir=".", output_path="f1_heatmap.png"):
    all_results = []
    for exp_name in experiment_names:
        path = os.path.join(results_dir, f"results_{exp_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["experiment"] = exp_name
            all_results.append(df)

    if not all_results:
        print("No result files found to generate heatmap.")
        return

    df_all = pd.concat(all_results)
    pivot = df_all.pivot(index="experiment", columns="model", values="f1_val")

    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("F1 Score at default Threshold by Experiment and Model")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"F1 score heatmap saved to {output_path}")


def plot_f1_heatmap_best_threshold(experiment_names, results_dir=".", output_path="f1_heatmap_best_threshold.png"):

    all_results = []
    for exp_name in experiment_names:
        path = os.path.join(results_dir, f"results_{exp_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["experiment"] = exp_name
            all_results.append(df)

    if not all_results:
        print("No result files found to generate heatmap.")
        return

    df_all = pd.concat(all_results)

    if "f1_val_at_best_threshold" not in df_all.columns:
        print("Column 'f1_val_best_threshold' not found in result files.")
        return

    pivot = df_all.pivot(index="experiment", columns="model", values="f1_val_at_best_threshold")

    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("F1 Score at Best Threshold by Experiment and Model")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"F1 score (best threshold) heatmap saved to {output_path}")


def save_classification_report(y_true, y_pred, model_name, experiment_name, suffix="default", output_dir="reports"):
    """
    Save the classification report to a text file.
    """
    report = classification_report(y_true, y_pred, digits=4)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{experiment_name}_{model_name}_classification_report_{suffix}.txt")

    with open(report_path, "w") as f:
        f.write(report)
