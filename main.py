'''
Code to run the REST API for the Census Data Salaray prediction.
'''
import pickle
from typing import Any, List, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data

app = FastAPI()


CAT_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


@app.on_event("startup")
async def startup_event():
    global MODEL, ENCODER, LB
    MODEL = pickle.load(open('./model/random_forest.pickle', 'rb'))
    ENCODER = pickle.load(open('./model/encoder.pickle', 'rb'))
    LB = pickle.load(open('./model/labelbinarizer.pickle', 'rb'))


class ResponseItem(BaseModel):
    status_code: int
    predictions: List[Union[str, None]] = []


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias=' education-num')
    marital_status: str = Field(alias=' marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias=' capital-gain')
    capital_loss: int = Field(alias=' capital-loss')
    hours_per_week: int = Field(alias=' hours-per-week')
    native_country: str = Field(alias=' native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                " education-num": 13,
                " marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                " capital-gain": 2174,
                " capital-loss": 0,
                " hours-per-week": 40,
                " native-country": "United-States"
            }

        }


@app.get("/")
async def root():
    return {"message": "Welcome to the API for "
                       "census data based salary prediction."}


@app.post("/predict", response_model=ResponseItem)
async def predict(input: CensusData) -> Any:
    try:
        x = pd.DataFrame([dict(input)])
        x_preprocessed, y, _, _ = \
            process_data(x,
                         categorical_features=CAT_FEATURES,
                         label=None,
                         training=False,
                         encoder=ENCODER,
                         lb=LB
                         )
        pred = MODEL.predict(x_preprocessed)
        predictions = LB.inverse_transform(pred)
        return ResponseItem(status_code=200, predictions=predictions.tolist())
    except Exception as e:
        print('Exception', e)
        return ResponseItem(status_code=503, predictions=[None])
