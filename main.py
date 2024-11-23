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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Prediction API"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([input_data.dict()])
    prediction = model.predict(data)[0]
    return {"predicted_insurance_cost": prediction}
