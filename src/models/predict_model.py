import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix
import joblib
import json

def load_dataset(url :str) -> pd.DataFrame:
    x_test=pd.read_csv(url)
    return x_test
    
def predict(df : pd.DataFrame) :
    model=joblib.load(r'artifict\model\random_forest.joblib')
    print('model_loaded')
    df=df.drop(columns='loan_approved').values
    y_pred=model.predict(df)
    print('y_pred generated')
    return y_pred

def eval_metric(y_pred,y_test):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("✅ Model trained successfully!")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    metrics = {
    "accuracy": acc,
    "confusion_matrix": cm.tolist(),
    "classification_report": report
    }

    
    with open(r"artifict\model\report.json", "wb") as f:
        json.dump(metrics, f, indent=4)

        print("✅ Model and metrics saved in 'artifacts/' directory.")
    

def main():
    x_test=load_dataset(r'C:\Users\mdmuz\OneDrive\Desktop\Data Science\mlops_project\loan_approve_using-coockiecutter\Loan_approval_using_cookiecutter\data\interim\x_test.csv')
    print('dataset loaded')
    y_pred=predict(x_test)
    print('prdiction is done')
    eval_metric(y_pred,x_test.iloc[:,-1])
    print('eval metrics')
    
if __name__ == "__main__":
    main()
    