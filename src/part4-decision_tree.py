'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def decision_tree(df_arrests):
    # Check if required columns exist
    if 'pred_universe' not in df_arrests.columns or 'num_fel_arrests_last_year' not in df_arrests.columns:
        print("Error: Required columns are missing from the dataframe.")
        return
    
    X = df_arrests[['pred_universe', 'num_fel_arrests_last_year']]
    y = df_arrests['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
    )

    # Define parameter grid for GridSearchCV
    param_grid_dt = {'max_depth': [3, 5, 7]}
    dt_model = DecisionTreeClassifier()

    # Initialize GridSearchCV
    gs_cv_dt = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid_dt,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit GridSearchCV
    gs_cv_dt.fit(X_train, y_train)

    # Optimal max_depth value and regularization
    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"What was the optimal value for max_depth? {optimal_max_depth}")

    if optimal_max_depth == max(param_grid_dt['max_depth']):
        print("The optimal max_depth has the least regularization.")
    elif optimal_max_depth == min(param_grid_dt['max_depth']):
        print("The optimal max_depth has the most regularization.")
    else:
        print("The optimal max_depth is in the middle range of regularization.")

    # Predict on the test set
    y_pred = gs_cv_dt.predict(X_test)

    # Add predictions to the dataframe
    df_arrests.loc[X_test.index, 'pred_dt'] = y_pred

    # Print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Save results to CSV
    df_arrests.to_csv('./data/df_arrests_with_dt_predictions.csv', index=False)

    return df_arrests

