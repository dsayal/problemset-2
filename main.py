"""
main.py

This script is the entry point for running the problem set. It orchestrates the entire data processing workflow by calling functions from various modules. The script performs the following steps:

1. **ETL Process**: Executes the ETL (Extract, Transform, Load) operations to prepare and save datasets in the `./data/` directory.
2. **Data Preprocessing**: Processes the raw data to prepare it for analysis.
3. **Logistic Regression**: Applies logistic regression to the preprocessed data.
4. **Decision Tree**: Applies a decision tree model to the data.
5. **Calibration Plot**: Generates calibration plots to evaluate model performance.

Modules:
- `etl`: Contains functions for data extraction, transformation, and loading.
- `preprocessing`: Provides functions for data cleaning and preparation.
- `logistic_regression`: Implements logistic regression modeling.
- `decision_tree`: Implements decision tree modeling.
- `calibration_plot`: Generates calibration plots for model evaluation.
"""

import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot

def main():
    """
    Main function to execute the ETL, preprocessing, modeling, and evaluation steps.
    """
    # PART 1: Instantiate ETL, saving the two datasets in `./data/`
    etl.etl()
    
    # PART 2: Call functions from preprocessing
    df_arrests = preprocessing.preprocessing()
    
    # PART 3: Call functions from logistic_regression
    df_arrests = logistic_regression.logistic_regression(df_arrests)
    
    # PART 4: Call functions from decision_tree
    df_arrests = decision_tree.decision_tree(df_arrests)
    
    # PART 5: Call functions from calibration_plot
    calibration_plot.calibration_plot(df_arrests)

if __name__ == "__main__":
    main()


