import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
def load_dataset(url:str) -> pd.DataFrame:
    x_train =pd.read_csv(url)
    y_train =x_train.iloc[:,-1]
    return x_train.drop(columns="loan_approved"),y_train

def model_building(x_train :pd.DataFrame,y_train:pd.DataFrame):
    clf=RandomForestClassifier(n_estimators=50)
    clf.fit(x_train,y_train)
    path=os.path.join('artifict')
    os.makedirs(path,exist_ok=True)
    model_path = os.path.join(path, "random_forest.joblib")  
    joblib.dump(clf, model_path)
    
    
def main():
    print("before data set")
    x_train,y_train=load_dataset(r'data\interim\x_train.csv')
    print('data_set_loaded')
    model_building(x_train,y_train)
    
if __name__ =="__main__":
    main()
    



