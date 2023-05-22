import os
import pickle
import subprocess

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from ml.data import clean_data
from ml.model import compute_model_metrics


@pytest.fixture
def data():
    data = pd.read_csv('./data/census.csv')
    return data


@pytest.fixture
def model():
    return pickle.load(open('./model/random_forest.pickle', 'rb'))


@pytest.fixture
def client():
    return TestClient(app)


def test_get(client):
    """
    Test get method on root of API.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the API for "
                   "census data based salary prediction."}


def test_post_lessthan50k(client):
    """
    Test inference on model via API for case less than 50k.
    """

    inputs = [{'age': 48, 'workclass': 'Private', 'fnlgt': 45612,
               'education': 'HS-grad',
               ' education-num': 9, ' marital-status': 'Never-married',
               'occupation': 'Adm-clerical',
               'relationship': 'Unmarried', 'race': 'Black',
               'sex': 'Female', ' capital-gain': 0, ' capital-loss': 0,
               ' hours-per-week': 37, ' native-country': 'United-States'},
              ]
    results = ['<=50K']
    for i in range(len(inputs)):
        response = client.post("/predict", json=inputs[i])
        assert response.json()['status_code'] == 200
        assert response.json()['predictions'] == [results[i]]


def test_post_greater50k(client):
    """
    Test inference on model via API.
    """

    inputs = [{'age': 51, 'workclass': 'Private', 'fnlgt': 83311,
               'education': 'Masters',
               ' education-num': 14, ' marital-status': 'Married-civ-spouse',
               'occupation': 'Craft-repair',
               'relationship': 'Husband', 'race': 'White', 'sex': 'Male',
               ' capital-gain': 0, ' capital-loss': 0,
               ' hours-per-week': 40, ' native-country': 'United-States'},
              ]
    results = ['>50K']
    for i in range(len(inputs)):
        response = client.post("/predict", json=inputs[i])
        assert response.json()['status_code'] == 200
        assert response.json()['predictions'] == [results[i]]


def test_clean_data(data):
    """
    Test clean data removing the spaces from the columnnames.
    data: pd.DataFrame with raw input data
    """
    originalcolumns = data.columns.tolist()
    data = clean_data(data)
    assert sorted(originalcolumns) != sorted(data.columns.tolist())


def test_model_metrics():
    """
    Test metrics on sample data.
    """
    groundtruth = [1, 1, 1, 1, 0, 0, 0, 0]
    preds = [1, 1, 1, 0, 0, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(groundtruth, preds)
    assert precision == 0.75
    assert recall == 0.75
    assert fbeta == 0.75


def test_store_model():
    """
    Test whether model is stored after training.
    """
    filename = './model/random_forest.pickle'
    os.remove(filename)
    filename = './model/labelbinarizer.pickle'
    os.remove(filename)
    filename = './model/encoder.pickle'
    os.remove(filename)
    subprocess.run(["python3", "train_model.py"])
    print(os.listdir('./model'))
    assert 'random_forest.pickle' in os.listdir('./model')
    assert 'labelbinarizer.pickle' in os.listdir('./model')
    assert 'encoder.pickle' in os.listdir('./model')
