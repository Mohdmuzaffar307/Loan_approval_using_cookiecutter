import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data():
    df=pd.read_csv(r'C:\Users\mdmuz\OneDrive\Desktop\Data Science\mlops_project\loan_approve_using-coockiecutter\Loan_approval_using_cookiecutter\data\raw\loan_approve.csv')
    return df

def data_cleaning(df : pd.DataFrame)-> pd.DataFrame:
    df.drop(columns=["name"],inplace=True)
    df['loan_approved']=df['loan_approved'].map({False:0,True:1})
    return df

def save_data(path : str,x_train :pd.DataFrame,x_test : pd.DataFrame) -> None:
    path =os.path.join('data','processed')
    os.makedirs(path)
    x_train.to_csv(os.path.join(path,"x_train.csv"))
    x_test.to_csv(os.path.join(path,"x_test.csv"))
    
def main():
    raw_df=load_data()
    clean_df=data_cleaning(raw_df)
    x_train,x_test=train_test_split(clean_df,test_size=0.2,random_state=42)
    path=os.path.join('data','processed')
    save_data(path,x_train,x_test)
    
    
if __name__ == "__main__":
    main()
    
    