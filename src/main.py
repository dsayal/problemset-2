'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot

def main():
    # PART 1: Instantiate ETL and save the two datasets in `./data/`
    etl_instance = etl.ETL() 
    etl_instance.run() 

    # Load datasets
    df_arrests_test_lr = pd.read_csv('./data/df_arrests_test_lr.csv')
    df_arrests_test_dt = pd.read_csv('./data/df_arrests_test_dt.csv')

    # PART 2: Call functions/instantiate objects from preprocessing
    preprocessing_instance = preprocessing.Preprocessing()  # Assumes Preprocessing is a class
    df_arrests_test_lr = preprocessing_instance.process(df_arrests_test_lr)
    df_arrests_test_dt = preprocessing_instance.process(df_arrests_test_dt)

    # PART 3: Call functions/instantiate objects from logistic_regression
    lr_model = logistic_regression.LogisticRegressionModel()  # Assumes LogisticRegressionModel is a class
    lr_model.train(df_arrests_test_lr)  # Assumes there's a method to train the model
    lr_predictions = lr_model.predict(df_arrests_test_lr)  # Assumes there's a method to make predictions

    # PART 4: Call functions/instantiate objects from decision_tree
    dt_model = decision_tree.DecisionTreeModel()  # Assumes DecisionTreeModel is a class
    dt_model.train(df_arrests_test_dt)  # Assumes there's a method to train the model
    dt_predictions = dt_model.predict(df_arrests_test_dt)  # Assumes there's a method to make predictions

    # PART 5: Call functions/instantiate objects from calibration_plot
    # Create calibration plots
    calibration_plot.calibration_plot(df_arrests_test_lr['y'], lr_predictions, n_bins=5)
    calibration_plot.calibration_plot(df_arrests_test_dt['y'], dt_predictions, n_bins=5)

    # Compute and print metrics
    from sklearn.metrics import precision_score, roc_auc_score

    # Logistic Regression PPV
    top_50_lr = df_arrests_test_lr.nlargest(50, 'pred_lr')
    ppv_lr = precision_score(top_50_lr['y'], (top_50_lr['pred_lr'] > 0.5).astype(int))
    print(f"PPV for logistic regression model in the top 50: {ppv_lr:.4f}")

    # Decision Tree PPV
    top_50_dt = df_arrests_test_dt.nlargest(50, 'pred_dt')
    ppv_dt = precision_score(top_50_dt['y'], (top_50_dt['pred_dt'] > 0.5).astype(int))
    print(f"PPV for decision tree model in the top 50: {ppv_dt:.4f}")

    # Logistic Regression AUC
    auc_lr = roc_auc_score(df_arrests_test_lr['y'], df_arrests_test_lr['pred_lr'])
    print(f"AUC for logistic regression model: {auc_lr:.4f}")

    # Decision Tree AUC
    auc_dt = roc_auc_score(df_arrests_test_dt['y'], df_arrests_test_dt['pred_dt'])
    print(f"AUC for decision tree model: {auc_dt:.4f}")

    # Comparing models
    if auc_lr > auc_dt:
        print("Logistic Regression is more accurate based on AUC.")
    elif auc_dt > auc_lr:
        print("Decision Tree is more accurate based on AUC.")
    else:
        print("Both models have the same accuracy based on AUC.")

if __name__ == "__main__":
    main()
