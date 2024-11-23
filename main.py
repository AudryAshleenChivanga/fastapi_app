import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int

def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
