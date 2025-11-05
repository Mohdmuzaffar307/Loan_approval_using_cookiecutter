import  numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_train_dataset(url : str)-> pd.DataFrame:
    x_train =pd.read_csv(url)
    return x_train

def load_test_dataset(url : str)-> pd.DataFrame:
    x_test =pd.read_csv(url)
    return x_test





def normalization(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Separate categorical and numerical columns
    cat_cols = x_train.select_dtypes(include=['object', 'category']).columns
    num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

    # One-hot encode categorical columns
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    ohe.fit(x_train[cat_cols])

    x_train_cat = pd.DataFrame(ohe.transform(x_train[cat_cols]), 
                               columns=ohe.get_feature_names_out(cat_cols),
                               index=x_train.index)
    x_test_cat = pd.DataFrame(ohe.transform(x_test[cat_cols]), 
                              columns=ohe.get_feature_names_out(cat_cols),
                              index=x_test.index)

    # Scale numerical columns
    scaler = StandardScaler()
    scaler.fit(x_train[num_cols])

    x_train_num = pd.DataFrame(scaler.transform(x_train[num_cols]), 
                               columns=num_cols,
                               index=x_train.index)
    x_test_num = pd.DataFrame(scaler.transform(x_test[num_cols]), 
                              columns=num_cols,
                              index=x_test.index)

    # Combine encoded categorical and scaled numerical columns
    x_train_final = pd.concat([x_train_num, x_train_cat], axis=1)
    x_test_final = pd.concat([x_test_num, x_test_cat], axis=1)

    return x_train_final, x_test_final



def save_data(path : str,x_train_final : pd.DataFrame,x_test_final : pd.DataFrame) ->None:
    path =os.path.join(path)
    os.makedirs(path)
    x_train_final.to_csv(os.path.join(path,"x_train.csv"))
    x_test_final.to_csv(os.path.join(path,"x_test.csv"))
    
def main()->None:
    # print('datasetfn')
    x_train=load_train_dataset(r'C:\Users\mdmuz\OneDrive\Desktop\Data Science\mlops_project\loan_approve_using-coockiecutter\Loan_approval_using_cookiecutter\data\processed\x_train.csv')
    # print('x_train_datasetloaded')
    x_test=load_test_dataset(r'C:\Users\mdmuz\OneDrive\Desktop\Data Science\mlops_project\loan_approve_using-coockiecutter\Loan_approval_using_cookiecutter\data\processed\x_test.csv')
    # print('x_test_datasetloaded')
    x_train_final,x_test_final=normalization(x_train,x_test)
    # print('normalization done')
    path=os.path.join('data','interim')
    save_data(path,x_train_final,x_test_final)
    
    
    
if __name__ =="__main__":
    main()
