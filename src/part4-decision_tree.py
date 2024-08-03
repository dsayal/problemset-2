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
    X = df_arrests[['pred_universe', 'num_fel_arrests_last_year']]
    y = df_arrests['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
    )

    param_grid = {'max_depth': [3, 5, 7]}
    dt_model = DecisionTreeClassifier()

    gs_cv = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    gs_cv.fit(X_train, y_train)

    optimal_max_depth = gs_cv.best_params_['max_depth']
    print(f"What was the optimal value for max_depth? {optimal_max_depth}")

    y_pred = gs_cv.predict(X_test)
    df_arrests.loc[X_test.index, 'pred_dt'] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")

    df_arrests.to_csv('./data/df_arrests_with_dt_predictions.csv', index=False)

    return df_arrests
