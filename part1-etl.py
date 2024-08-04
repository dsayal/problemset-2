'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

import pandas as pd
import os

def etl():
    """This function performs the ETL (Extract, Transform, Load) process on two datasets and saves the processed datasets as CSV files in the `data/` directory.

Steps:
1. **Create Directory**: Ensures that the `data/` directory exists; if not, it creates the directory.
2. **Load Data**: Downloads two datasets in Feather format from specified URLs.
3. **Transform Data**:
   - Converts the `filing_date` column in both datasets to datetime format.
   - Renames the transformed date columns to `arrest_date_univ` for the prediction universe dataset and `arrest_date_event` for the arrest events dataset.
   - Drops the original `filing_date` column from both datasets.
4. **Load Data**: Saves the transformed datasets as CSV files in the `data/` directory:
   - `pred_universe_raw.csv` for the prediction universe dataset.
   - `arrest_events_raw.csv` for the arrest events dataset.
"""

    os.makedirs('./data', exist_ok=True)

    # Load data
    pred_universe_raw = pd.read_feather('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_feather('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
    
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])
    
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Save data to CSV
    pred_universe_raw.to_csv('./data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('./data/arrest_events_raw.csv', index=False)
    
    print("ETL process completed and data saved.")


