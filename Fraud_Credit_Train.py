import pickle
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from credit_fraud_data_utils import load_data, apply_scaling, apply_sampling, save_scaler
from credit_fraud_val_utils import evaluate_model, find_best_threshold, save_model, save_confusion_matrix, \
    plot_f1_heatmap, save_classification_report, plot_f1_heatmap_best_threshold

# Set the acceptable gap (delta) between train and val F1
DELTA = 0.01  # 3% gap allowed
EXPERIMENTS = [
    {"scaler": None, "sampling": None, "cost_sensitive": False, "include_mlp": False, "name": "raw"},
    {"scaler": "standard", "sampling": None, "cost_sensitive": False, "include_mlp": True, "name": "scaled"},
    {"scaler": None, "sampling": "undersample", "cost_sensitive": False, "include_mlp": False, "name": "undersample"},
    {"scaler": "standard", "sampling": "undersample", "cost_sensitive": False, "include_mlp": True,
     "name": "undersample_scaled"},
    {"scaler": None, "sampling": "oversample", "cost_sensitive": False, "include_mlp": False, "name": "oversample"},
    {"scaler": "standard", "sampling": "oversample", "cost_sensitive": False, "include_mlp": True,
     "name": "oversample_scaled"},
    {"scaler": None, "sampling": "over_and_under", "cost_sensitive": False, "include_mlp": False,
     "name": "over_and_under"},
    {"scaler": "standard", "sampling": "over_and_under", "cost_sensitive": False, "include_mlp": True,
     "name": "over_and_under_scaled"},
    {"scaler": None, "sampling": None, "cost_sensitive": True, "include_mlp": False, "name": "cost_sensitive"},
    {"scaler": "standard", "sampling": None, "cost_sensitive": True, "include_mlp": True,
     "name": "cost_sensitive_scaled"},
]


# def get_models(cost_sensitive=False, ratio=None, include_mlp=True):
#     # ... same as your original function ...
#     Lr = LogisticRegression(
#         solver='lbfgs',
#         class_weight={0: 1, 1: ratio} if cost_sensitive and ratio else 'balanced',
#         max_iter=1500,
#         random_state=42,
#     )
#     rf = RandomForestClassifier(
#         max_depth=12,
#         n_estimators=50,
#         random_state=42,
#         min_samples_split=5,
#         min_samples_leaf=8,
#         max_features='log2',
#         class_weight={0: 1, 1: ratio} if cost_sensitive and ratio else None
#     )
#
#     mlp = MLPClassifier(
#         hidden_layer_sizes=(50,),
#         max_iter=300,
#         alpha=0.001,
#         learning_rate_init=0.005,
#         random_state=42,
#         solver='adam',
#         learning_rate='adaptive',
#         batch_size=32,
#         activation='tanh',
#         early_stopping=True,
#         n_iter_no_change=10,
#     )
#
#     models = {
#         "LogisticRegression": Lr,
#         "RandomForest": rf,
#     }
#
#     if include_mlp:
#         models["MLPClassifier"] = mlp
#
#     voting_clf = VotingClassifier(
#         estimators=[(k.lower(), v) for k, v in models.items()],
#         voting='soft'
#     )
#     models["VotingClassifier"] = voting_clf
#     return models

def get_models(cost_sensitive=False, ratio=None, include_mlp=True):
    Lr = LogisticRegression(
        solver='lbfgs',
        class_weight={0: 1, 1: ratio} if cost_sensitive and ratio else 'balanced',
        max_iter=1500,
        random_state=42,
    )
    rf = RandomForestClassifier(
        max_depth=12,
        n_estimators=50,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=8,
        max_features='log2',
        class_weight={0: 1, 1: ratio} if cost_sensitive and ratio else None
    )
    # xgb = XGBClassifier(
    #     use_label_encoder=False,
    #     eval_metric='aucpr',
    #     random_state=42,
    #     max_depth=5,
    #     n_estimators=400,
    #     learning_rate=0.1,
    #     subsample=0.6,
    #     colsample_bytree=1.0,
    #     reg_lambda=1.0,
    #     gamma=0.2,
    #     scale_pos_weight=ratio/2 if cost_sensitive and ratio else 1
    # )
    # catboost = CatBoostClassifier(
    #     iterations=250,
    #     learning_rate=0.1,
    #     depth=4,
    #     eval_metric='AUC',
    #     verbose=0,
    #     l2_leaf_reg=3.0,
    #     random_state=42,
    #     scale_pos_weight=ratio if cost_sensitive and ratio else None
    # )
    mlp = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=300,
        alpha=0.001,
        learning_rate_init=0.005,
        random_state=42,
        solver='adam',
        learning_rate='adaptive',
        batch_size=32,
        activation='tanh',
        early_stopping=True,
        n_iter_no_change=10,
    )
    models = {
        "LogisticRegression": Lr,
        "RandomForest": rf,
        # "XGBoost": xgb,
        # "CatBoost": catboost
    }
    if include_mlp:
        models["MLPClassifier"] = mlp
    voting_clf = VotingClassifier(
        estimators=[(k.lower(), v) for k, v in models.items()],
        voting='soft'
    )
    models["VotingClassifier"] = voting_clf
    return models


def run_experiment(config, x_train, y_train, x_val, y_val):
    scaler = None
    if config["scaler"]:
        x_train, x_val, _, scaler = apply_scaling(x_train, x_val, None, config["scaler"])
        save_scaler(scaler, f"{config['name']}_scaler.pkl")

    if config["sampling"]:
        x_train, y_train = apply_sampling(x_train, y_train, config["sampling"])

    ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    models = get_models(cost_sensitive=config["cost_sensitive"], ratio=ratio, include_mlp=config["include_mlp"])

    results = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        # Evaluate on train and val
        f1_val, pr_auc_val, thresholds, y_val_proba, y_val_pred, y_train_pred, y_train_proba, f1_train = evaluate_model(
            model, name, x_train, y_train, x_val, y_val
        )

        threshold, f1_at_best = find_best_threshold(y_train, y_train_proba)

        y_val_pred_best = (y_val_proba >= threshold).astype(int)
        f1_val_at_best = f1_score(y_val, y_val_pred_best)

        y_train_pred_best = (y_train_proba >= threshold).astype(int)
        f1_train_at_best = f1_score(y_train, y_train_pred_best)

        # Save classification reports for both thresholds
        save_classification_report(y_train, y_train_pred, model_name=name + "_train",
                                   experiment_name=config["name"], suffix="default_thresh")
        save_classification_report(y_val, y_val_pred, model_name=name + "_val",
                                   experiment_name=config["name"], suffix="default_thresh")
        save_classification_report(y_train, y_train_pred_best, model_name=name + "_train",
                                   experiment_name=config["name"], suffix="best_thresh")
        save_classification_report(y_val, y_val_pred_best, model_name=name + "_val",
                                   experiment_name=config["name"], suffix="best_thresh")

        # Save confusion matrices
        save_confusion_matrix(y_val, y_val_pred, model_name=name,
                              experiment_name=config["name"], suffix="default")
        save_confusion_matrix(y_val, y_val_pred_best, model_name=name + "_best_thresh",
                              experiment_name=config["name"], suffix="best_thresh")

        results.append({
            "model": name,
            "f1_train": f1_train,
            "f1_val": f1_val,
            "f1_train_at_best_threshold": f1_train_at_best,
            "f1_val_at_best_threshold": f1_val_at_best,
            "pr_auc_val": pr_auc_val,
            "best_threshold": threshold
        })

        # Save models with both thresholds
        model_path_default = f"model_{config['name']}_{name}_default_thresh.pkl"
        save_model(model, scaler, 0.5, name, model_path_default, config=config)

        model_path_best = f"model_{config['name']}_{name}_best_thresh.pkl"
        save_model(model, scaler, threshold, name, model_path_best, config=config)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_{config['name']}.csv", index=False)
    print(f"Results saved to results_{config['name']}.csv")

    # Plot F1 scores
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1_val"])
    plt.ylabel("F1 Score (Validation)")
    plt.title(f"F1 Score by Model ({config['name']})")
    plt.tight_layout()
    plt.savefig(f"f1_scores_{config['name']}.png")
    plt.close()
    print(f"F1 score plot saved to f1_scores_{config['name']}.png")


def main():
    parser = argparse.ArgumentParser(description='Train credit fraud detection models')
    parser.add_argument('--train_path', type=str,
                        default=r'D:\programming\ML\bianry classification\project\train.csv',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str,
                        default=r'D:\programming\ML\bianry classification\project\val.csv',
                        help='Path to validation data')
    parser.add_argument('--run', type=str, default="all",
                        help="all or best. If best, runs only the best experiment.")

    args = parser.parse_args()

    # Load data
    x_train, y_train, x_val, y_val = load_data(args.train_path, args.val_path)

    if args.run == "all":
        for config in EXPERIMENTS:
            print(f"\nRunning experiment: {config['name']}")
            run_experiment(config, x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy())

        # Generate comparison plots
        plot_f1_heatmap([cfg['name'] for cfg in EXPERIMENTS])
        plot_f1_heatmap_best_threshold([cfg['name'] for cfg in EXPERIMENTS])

    elif args.run == "best":
        best_config = EXPERIMENTS[1]  # Example: the second config is the best
        print(f"\nRunning best experiment: {best_config['name']}")
        run_experiment(best_config, x_train, y_train, x_val, y_val)
    else:
        print("Unknown --run option. Use 'all' or 'best'.")


if __name__ == "__main__":
    main()
