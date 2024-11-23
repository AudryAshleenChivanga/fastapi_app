import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_filename = 'best_model.pkl'

class PredictionInput(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int

def load_model():
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

@app.post("/predict")
async def predict(input_data: PredictionInput):
    model = load_model()
    data = [[input_data.age, input_data.sex, input_data.bmi, input_data.children, input_data.smoker, input_data.region]]
    
    prediction = model.predict(data)
    return {"predicted_charge": prediction[0]}

