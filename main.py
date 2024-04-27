import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
import mlflow
from mlflow.models import infer_signature

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
    print("The report for prediction",report)

   # infer model signature
    train_pred = rc.predict(train_x)
    signature = infer_signature(train_x, train_pred)

    # Log mlflow attributes for mlflow UI
    metrics = {
        "classification_report": report
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rc, "model", signature=signature)
