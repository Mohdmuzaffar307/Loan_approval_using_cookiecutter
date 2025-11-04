import numpy as np
import pandas as pd
import os

def load_data(url : str) -> pd.DataFrame:
    df=pd.read_csv(url)
    return df

def save_raw_data(path : str,df : pd.DataFrame)-> None:
    path=os.path.join(path)
    os.makedirs(path,exist_ok=True)
    df.to_csv(os.path.join(path,'loan_approve.csv'))


    

def main():
    url = r"C:/Users/mdmuz/.cache/kagglehub/datasets/anishdevedward/loan-approval-dataset\versions/1/loan_approval.csv"
    df=load_data(url)
    path =os.path.join('data','raw')
    save_raw_data(path,df)

if __name__ == "__main__":
    main()

    
