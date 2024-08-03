'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    # Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

# Load data
df_arrests_test_lr = pd.read_csv('./data/df_arrests_test_lr.csv')
df_arrests_test_dt = pd.read_csv('./data/df_arrests_test_dt.csv')

# Extract y_true
y_test_lr = df_arrests_test_lr['y']
y_test_dt = df_arrests_test_dt['y']

# Plot calibration curves
calibration_plot(y_test_lr, df_arrests_test_lr['pred_lr'], n_bins=5)
calibration_plot(y_test_dt, df_arrests_test_dt['pred_dt'], n_bins=5)

# Convert predicted probabilities to binary outcomes
threshold = 0.5
y_pred_lr = (df_arrests_test_lr['pred_lr'] > threshold).astype(int)
y_pred_dt = (df_arrests_test_dt['pred_dt'] > threshold).astype(int)

# Extra Credit Calculations
# Logistic Regression PPV
top_50_lr = df_arrests_test_lr.nlargest(50, 'pred_lr')
ppv_lr = precision_score(top_50_lr['y'], (top_50_lr['pred_lr'] > threshold).astype(int))
print(f"PPV for logistic regression model in the top 50: {ppv_lr:.4f}")

# Decision Tree PPV
top_50_dt = df_arrests_test_dt.nlargest(50, 'pred_dt')
ppv_dt = precision_score(top_50_dt['y'], (top_50_dt['pred_dt'] > threshold).astype(int))
print(f"PPV for decision tree model in the top 50: {ppv_dt:.4f}")

# Logistic Regression AUC
auc_lr = roc_auc_score(y_test_lr, df_arrests_test_lr['pred_lr'])
print(f"AUC for logistic regression model: {auc_lr:.4f}")

# Decision Tree AUC
auc_dt = roc_auc_score(y_test_dt, df_arrests_test_dt['pred_dt'])
print(f"AUC for decision tree model: {auc_dt:.4f}")

# Comparing models
if auc_lr > auc_dt:
    print("Logistic Regression is more accurate based on AUC.")
elif auc_dt > auc_lr:
    print("Decision Tree is more accurate based on AUC.")
else:
    print("Both models have the same accuracy based on AUC.")