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
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl()
    
    # PART 2: Call functions from preprocessing
    df_arrests = preprocessing()
    
    # PART 3: Call functions from logistic_regression
    logistic_regression(df_arrests)
    
    # PART 4: Call functions from decision_tree
    decision_tree(df_arrests)
    
    # PART 5: Call functions from calibration_plot
    calibration_plot(df_arrests)

if __name__ == "__main__":
    main()