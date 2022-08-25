from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib


def preprocess(X, y):
    """
    Preprocess data.
    """
    pipeline = make_pipeline(SimpleImputer(), StandardScaler())

    X_preprocessed = pipeline.fit_transform(X)

    # Save the pipeline to disk.
    joblib.dump(pipeline, 'models/preprocess.pkl')
    # Save the preprocessed data to disk as a DataFrame
    df = pd.DataFrame(X_preprocessed, columns=X.columns)
    df['class'] = y
    df.to_csv('data/data_processed.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')
    preprocess(data.drop('class', axis=1), data[['class']])
