'''
You will run this problem set from main.py, so set things up accordingly
'''
import etl
import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve


def main():
    # PART 1: Instantiate ETL, saving the two datasets in `./data/`
    etl()
    
    # PART 2: Call functions from preprocessing
    df_arrests = preprocessing()
    
    # PART 3: Call functions from logistic_regression
    df_arrests = LogisticRegression(df_arrests)
    
    # PART 4: Call functions from decision_tree
    df_arrests = DecisionTreeClassifier(df_arrests)
    
    # PART 5: Call functions from calibration_plot
    calibration_curve(df_arrests)

if __name__ == "__main__":
    main()
