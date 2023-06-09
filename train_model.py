'''
Main script to run training of the
model for the Census Data Salaray prediction.
'''
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import clean_data, process_data
from ml.model import (compute_sliced_metrics, store_model,
                      train_model, compute_model_metrics)

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('./data/census.csv')

data = clean_data(data)
# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)
grouped = train.groupby('salary')
train = grouped.apply(
        lambda group: group.sample(grouped.size().min())
    ).reset_index(drop=True)


cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

X_train, y_train, encoder, lb = process_data(
    deepcopy(train),
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    deepcopy(test),
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

X_test = pd.DataFrame(X_test)
X_test['race'] = test['race']
precision, recall, fbeta = compute_model_metrics(y_pred, y_test)


compute_sliced_metrics(X_test, y_test, model, category='race')


store_model(encoder, './model/encoder.pickle')
store_model(lb, './model/labelbinarizer.pickle')
store_model(model, './model/random_forest.pickle')
