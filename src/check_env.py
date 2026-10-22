import pandas as pd
import sklearn
import xgboost
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("Rows, columns:", df.shape)
print(df.head(3))
