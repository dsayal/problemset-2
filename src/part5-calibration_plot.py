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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

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

    # Print calibration curves summary
    print("Calibration plots for Logistic Regression and Decision Tree have been generated.")
