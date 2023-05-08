'''
Implement model functionality for the
model for the Census Data Salaray prediction.
'''

import pickle

import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(n_estimators=400, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf


def compute_sliced_metrics(X, y, model, category='education'):
    """
    Compute metrics on data slices to provide
    insight if there is any bias.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    y: np.array
        True labels
    category: str
        The categorical feature on which to slice
        data for performance evaluation
    Returns
        None
    """
    bias_dict = dict()
    for name, group in X.groupby(category):
        subsetlabels = y[group.index]
        group = group.drop(category, axis=1)
        preds = inference(model, group.values)
        precision, recall, fbeta = compute_model_metrics(subsetlabels, preds)
        bias_dict[name + '_' + category] = [precision, recall, fbeta]

    with open('./data/sliced_output.txt', 'w') as file:
        file.write(json.dumps(bias_dict))  # use `json.loads` to do the reverse

    return bias_dict


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning
    model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def store_model(model, filename):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    filename : str
        Model location to store model at
    Returns
    -------
        None
    """
    pickle.dump(model, open(filename, 'wb'))
