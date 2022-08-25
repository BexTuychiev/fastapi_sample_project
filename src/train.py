from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd


def train():
    data = pd.read_csv('data/data_processed.csv')
    X = data.drop('class', axis=1)
    y = data['class']

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(X, y)

    # Save the model to disk.
    joblib.dump(clf, 'models/rf.pkl')


if __name__ == '__main__':
    train()
