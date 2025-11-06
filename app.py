from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
 
# Load the trained model
model = joblib.load("iris_model.pkl")
 
app = FastAPI()
 
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 
class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str
 
 
@app.get("/")
def read_root():
    return {"Hello": "World"}
 
@app.post("/predict", response_model=IrisPrediction)
def predict(data: IrisInput):
 
    # prepare input data
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
 
    # make prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]
    return IrisPrediction(predicted_class=predicted_class, predicted_class_name=predicted_class_name)
 
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.0", port=8000)