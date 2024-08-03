'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
# logistic_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogisticRegressionModel:
    def train(self, df_arrests):
        # Split the data into training and test sets
        X = df_arrests[['pred_universe', 'num_fel_arrests_last_year']]
        y = df_arrests['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
        )
        
        # Define the features
        features = ['pred_universe', 'num_fel_arrests_last_year']
        
        # Define the parameter grid for GridSearchCV
        param_grid = {'C': [0.1, 1, 10]}  # Example values for C, adjust as needed

        # Initialize the Logistic Regression model
        lr_model = LogisticRegression()

        # Initialize GridSearchCV with 5-fold cross-validation
        gs_cv = GridSearchCV(
            estimator=lr_model, 
            param_grid=param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit the GridSearchCV model
        gs_cv.fit(X_train, y_train)

        # Optimal value for C
        optimal_C = gs_cv.best_params_['C']
        print(f"What was the optimal value for C? {optimal_C}")
        
        # Print if the optimal C has the most, least or middle regularization
        if optimal_C == max(param_grid['C']):
            print("The optimal value for C has the least regularization.")
        elif optimal_C == min(param_grid['C']):
            print("The optimal value for C has the most regularization.")
        else:
            print("The optimal value for C is in the middle range of regularization.")

        # Predict on the test set
        y_pred = gs_cv.predict(X_test)
        df_arrests.loc[X_test.index, 'pred_lr'] = y_pred
        
        # Calculate and print accuracy for sanity check
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {accuracy:.4f}")

        # Save results to CSV for use in PART 4 and PART 5
        df_arrests.to_csv('./data/df_arrests_with_lr_predictions.csv', index=False)

        return X_test, y_test, df_arrests

if __name__ == "__main__":
    # Example usage for testing purposes
    df_arrests = pd.read_csv('./data/df_arrests.csv')
    lr_model = LogisticRegressionModel()
    X_test, y_test, df_arrests = lr_model.train(df_arrests)
    print("Training and predictions complete.")

