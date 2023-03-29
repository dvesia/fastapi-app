#import os
#from fastapi import FastAPI
#from pydantic import BaseModel
#from typing_extensions import Literal
#from joblib import load
#from pandas.core.frame import DataFrame
#from src import utils
#import numpy as np
#import traceback
#
#
#class User(BaseModel):
#    age: int
#    workclass: Literal[
#        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
#        'Local-gov', 'Self-emp-inc', 'Without-pay']
#    education: Literal[
#        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
#        'Some-college',
#        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
#        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
#    maritalStatus: Literal[
#        'Never-married', 'Married-civ-spouse', 'Divorced',
#        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
#        'Widowed']
#    occupation: Literal[
#        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
#        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
#        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
#        'Craft-repair', 'Protective-serv', 'Armed-Forces',
#        'Priv-house-serv']
#    relationship: Literal[
#        'Not-in-family', 'Husband', 'Wife', 'Own-child',
#        'Unmarried', 'Other-relative']
#    race: Literal[
#        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
#        'Other']
#    sex: Literal['Male', 'Female']
#    hoursPerWeek: int
#    nativeCountry: Literal[
#        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
#        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
#        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
#        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
#        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
#        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
#        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
#        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
#        'Holand-Netherlands']
#
#
#if "DYNO" in os.environ and os.path.isdir(".dvc"):
#    os.system("dvc config core.no_scm true")
#    if os.system("dvc pull") != 0:
#        exit("dvc pull failed")
#    os.system("rm -r .dvc .apt/usr/lib/dvc")
#
#app = FastAPI()
#
#
#@app.get("/")
#async def get_items():
#    return {"message": "Greetings!"}
#
#
#@app.post("/")
#async def inference(user_data: User):
#    try:
#        model = load("artifacts/models/model.joblib")
#        encoder = load("artifacts/models/encoder.joblib")
#        lb = load("artifacts/models/label_binarizer.joblib")
#
#        array = np.array([[
#            user_data.age,
#            user_data.workclass,
#            user_data.education,
#            user_data.maritalStatus,
#            user_data.occupation,
#            user_data.relationship,
#            user_data.race,
#            user_data.sex,
#            user_data.hoursPerWeek,
#            user_data.nativeCountry
#        ]])
#        df_temp = DataFrame(data=array, columns=[
#            "age",
#            "workclass",
#            "education",
#            "marital-status",
#            "occupation",
#            "relationship",
#            "race",
#            "sex",
#            "hours-per-week",
#            "native-country",
#        ])
#
#        X, _, _, _ = utils.process_data(
#            df_temp,
#            categorical_features=utils.get_cat_features(),
#            encoder=encoder, lb=lb, training=False)
#        pred = utils.inference(model, X)
#        y = lb.inverse_transform(pred)[0]
#        return {"prediction": y}
#    except Exception as e:
#        print(f"An error occurred: {str(e)}")
#        print(traceback.format_exc())
#        raise e
#

"""
This module contains the code for the API
"""
import os
from fastapi import FastAPI
from typing import Literal
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel
from src.utils import process_data, get_cat_features, inference
from joblib import load

# Set up DVC on Heroku
if os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc")


# Create app
app = FastAPI()

# POST Input Schema
class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": 27,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 35,
                "native_country": 'United-States'
            }
        }


# Load artifacts
model = load("artifacts/models/model.joblib")
encoder = load("artifacts/models/encoder.joblib")
lb = load("artifacts/models/label_binarizer.joblib")


# Root path
@app.get("/")
async def root():
    return {
        "Hi": "This app predicts wether income exceeds $50K/yr based on census data."}

# Prediction path
@app.post("/")
async def predict(input: ModelInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.education,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.hours_per_week,
        input.native_country]])

    original_cols = [
        "age",
        "workclass",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country"]

    input_df = DataFrame(data=input_data, columns=original_cols)
    cat_features = get_cat_features()

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)