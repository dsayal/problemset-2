'''
You will run this problem set from main.py, so set things up accordingly
'''
import pandas as pd
from part1_etl import etl
from part2_preprocessing import preprocessing
from part3_logistic_regression import logistic_regression
from part4_decision_tree import decision_tree
from part5_calibration_plot import calibration_plot

def main():
    # PART 1: Instantiate ETL, saving the two datasets in `./data/`
    etl()
    
    # PART 2: Call functions from preprocessing
    df_arrests = preprocessing()
    
    # PART 3: Call functions from logistic_regression
    df_arrests = logistic_regression(df_arrests)
    
    # PART 4: Call functions from decision_tree
    df_arrests = decision_tree(df_arrests)
    
    # PART 5: Call functions from calibration_plot
    calibration_plot(df_arrests)

if __name__ == "__main__":
    main()
