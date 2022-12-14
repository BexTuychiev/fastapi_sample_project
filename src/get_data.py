from sklearn.datasets import make_classification
import pandas as pd

n_samples = 10000
n_features = 10


def generate_data(n_samples=n_samples, n_features=n_features):
    """
    Generate a random binary classification dataset.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_features, n_redundant=0, n_classes=2,
                               n_clusters_per_class=1)
    cols = [f'feat_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['class'] = y
    df.to_csv('data/data.csv', index=False)


if __name__ == '__main__':
    generate_data()
