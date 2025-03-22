"""
Dataset Preprocessing

This script preprocesses the subject data based on the following steps:
1. Assign patients with exit of study as NA to values of 0 (assume survivor)
2. Select patients with Holter ECG available 
3. Keep only patients that were either survivors or died; exclude patients lost to follow-up or had cardiac transplantation
4. Exclude patients with non-cardiac deaths
5. Reassign pump failure values to only be 7 (originally a mix of 6 and 7)
6. Sort patients by cause of death, followed by patient ID

"""

import pandas as pd

def preprocessing(df = pd.DataFrame) -> None:
    """
    Preprocesses the dataset following the steps listed abovem and writes a new csv file called subject_info_cleaned.csv.

    Arguments:
        df (pd.DataFrame): The original subject data to preprocess.

    Returns:
        None
    """
    
    # Assign patients with exit of study as NA to values of 0 (survivor)
    df['Exit of the study'] = df['Exit of the study'].fillna(0)
    
    # Select patients with Holter ECG available
    df = df[df['Holter available'] == 1] # 992 -> 936
    
    # Keep only patients that were either survivors or died
    # Exclude patients lost to follow-up or had cardiac transplantation
    df = df[df['Exit of the study'].isin([0, 3])] # 936 -> 906
    
    # Exclude patients with non-cardiac deaths
    df = df[df['Cause of death'] != 1] # 906 -> 849
    
    # Reassign pump failure values to only be 7
    df['Cause of death'] = df['Cause of death'].replace(6, 7)
    
    # Sort based on cause of death
    # 0 = survivor, 3 = sudden cardiac death, 6 = pump failure death
    df = df.sort_values(by = ['Cause of death', 'Patient ID'])
    
    # Write csv
    df.to_csv('../Data/subject-info-cleaned.csv', index = False)
    

if __name__ == "__main__":

    # Read in dataset
    df = pd.read_csv("../Data/subject-info.csv", delimiter = ";")
    
    # Preprocess the data
    preprocessing(df)
    