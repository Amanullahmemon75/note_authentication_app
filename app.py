from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Load saved models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rfc_model = joblib.load('random_forest_model.pkl')
knn_model = joblib.load('knn_model.pkl')
neural_network_model = joblib.load('neural_network_model.pkl')

# Load scaler (assuming you saved it as scaler.pkl)
scaler = joblib.load('scaler.pkl')

# Jinja2 template renderer
templates = Jinja2Templates(directory="templates")

# Define function to make predictions
def make_prediction(model, data):
    prediction = model.predict([data])
    return prediction[0]

# Home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, variance: float = Form(...), skewness: float = Form(...), 
                  curtosis: float = Form(...), entropy: float = Form(...)):
    # Prepare the feature vector
    features = np.array([variance, skewness, curtosis, entropy])

    # Scale the input features
    scaled_features = scaler.transform([features])

    # Make predictions using each model
    logistic_regression_prediction = make_prediction(logistic_regression_model, scaled_features)
    svm_prediction = make_prediction(svm_model, scaled_features)
    rfc_prediction = make_prediction(rfc_model, scaled_features)
    knn_prediction = make_prediction(knn_model, scaled_features)
    nn_prediction = make_prediction(neural_network_model, scaled_features)

    # Render the results page with predictions
    return templates.TemplateResponse("result.html", {
        "request": request,
        "logistic_regression": logistic_regression_prediction,
        "svm": svm_prediction,
        "random_forest": rfc_prediction,
        "knn": knn_prediction,
        "neural_network": nn_prediction
    })
