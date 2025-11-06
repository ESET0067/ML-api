from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load("iris_model.pkl")

# Create FastAPI instance
app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")  # make sure folder name = templates

# Define input model (optional, for API-based prediction)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define output model (optional)
class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str


# ------------------------
# Root Route (Form Page)
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------
# Prediction Route
# ------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]

    # Render result template
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    )


# ------------------------
# For Local Testing
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
