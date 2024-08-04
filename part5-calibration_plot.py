import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

def calibration_plot(df_arrests):
    # Load the predictions
    y_true = df_arrests['y']
    y_prob_lr = df_arrests['pred_lr']
    y_prob_dt = df_arrests['pred_dt']

    def plot_calibration_curve(y_true, y_prob, model_name):
        bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=5)
        sns.set(style="whitegrid")
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(prob_true, bin_means, marker='o', label=f"{model_name} Model")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot for {model_name} Model")
        plt.legend(loc="best")
        plt.show()

    # Create calibration plots
    plot_calibration_curve(y_true, y_prob_lr, "Logistic Regression")
    plot_calibration_curve(y_true, y_prob_dt, "Decision Tree")

    # Calculate metrics for extra credit
    def compute_ppv(y_true, y_prob):
        # Get top 50 predicted risk scores
        top_50_idx = y_prob.nlargest(50).index
        y_true_top_50 = y_true.loc[top_50_idx]
        tp = y_true_top_50.sum()  # True Positives
        fp = len(y_true_top_50) - tp  # False Positives
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        return ppv

    ppv_lr = compute_ppv(y_true, y_prob_lr)
    ppv_dt = compute_ppv(y_true, y_prob_dt)
    print(f"PPV for Logistic Regression (Top 50): {ppv_lr:.4f}")
    print(f"PPV for Decision Tree (Top 50): {ppv_dt:.4f}")

    auc_lr = roc_auc_score(y_true, y_prob_lr)
    auc_dt = roc_auc_score(y_true, y_prob_dt)
    print(f"AUC for Logistic Regression: {auc_lr:.4f}")
    print(f"AUC for Decision Tree: {auc_dt:.4f}")

    # Compare models for calibration
    better_model = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
    print(f"Which model is more calibrated? The model with the higher AUC: {better_model}")

    # Assess if metrics agree on model accuracy
    if ppv_lr > ppv_dt:
        ppv_comparison = "Logistic Regression"
    elif ppv_dt > ppv_lr:
        ppv_comparison = "Decision Tree"
    else:
        ppv_comparison = "Both models have similar PPV"

    print(f"Do both metrics agree that one model is more accurate than the other? {ppv_comparison}")



