from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app with the name "Note Authentication App"
app = Flask(__name__, template_folder='templates')

# Load saved models (Make sure these files exist in your 'models' folder)
logistic_regression_model = joblib.load('models/logistic_regression_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
rfc_model = joblib.load('models/random_forest_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
neural_network_model = joblib.load('models/neural_network_model.pkl')

# Load scaler (assuming you saved it as scaler.pkl)
scaler = joblib.load('models/scaler.pkl')

# Define function to make predictions
def make_prediction(model, data):
    prediction = model.predict([data])
    return prediction[0]

# Home page route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get form input data
    variance = float(request.form['variance'])
    skewness = float(request.form['skewness'])
    curtosis = float(request.form['curtosis'])
    entropy = float(request.form['entropy'])

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

    # Inverse transform the scaled features to original values
    original_features = scaler.inverse_transform(scaled_features)

    # Render the results page with predictions and original data
    return render_template("result.html", 
                           logistic_regression=logistic_regression_prediction,
                           svm=svm_prediction,
                           random_forest=rfc_prediction,
                           knn=knn_prediction,
                           neural_network=nn_prediction,
                           original_features=original_features[0])

if __name__ == "__main__":
    app.run(debug=True)
