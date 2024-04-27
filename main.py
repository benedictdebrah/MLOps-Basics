from urllib.parse import urlparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

import mlflow.sklearn

# Load data
wine = datasets.load_wine()
X = wine.data
y = wine.target

# create pandas dataframe for data
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# evaluate metrics function
def evaluate_metrics(actual, predicted):
    report = classification_report(actual, predicted)
    return report

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #One must be commented **either** the local or the remote tracking URI
    
    # Set the local directory to log MLflow runs(uncomment the line below if you want to run on local)
    # mlflow.set_tracking_uri('file://' + os.path.abspath('./mlruns'))

    #Set the tracking URI based on the environment (uncomment this codes if you want to run on dagshub)
    remote_server_uri = "https://dagshub.com/bende.tymer/MLOps-Basics.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    # split the dataset
    train, test = train_test_split(df, test_size=0.2)

    train_x = train.drop('target', axis=1)
    train_y = train['target']
    test_x = test.drop('target', axis=1)
    test_y = test['target']

    # run model
    rc = RandomForestClassifier(random_state=42)
    rc.fit(train_x, train_y)
    predicted_ = rc.predict(test_x)

    # evaluate metrics
    report = evaluate_metrics(test_y, predicted_)
    print("The report for prediction", report)

    accuracy = accuracy_score(test_y, predicted_)
    precision = precision_score(test_y, predicted_, average='weighted')
    recall = recall_score(test_y, predicted_, average='weighted')
    f1 = f1_score(test_y, predicted_, average='weighted')

    # Log mlflow attributes for mlflow UI
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rc, "model")



