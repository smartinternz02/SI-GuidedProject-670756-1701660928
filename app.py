import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import catboost  # Importing catboost module

model = pickle.load(open('Airbnb_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio.html')
def portfolio():
    return render_template('portfolio.html')

@app.route('/pricing.html')
def pricing():
    return render_template('pricing.html')

@app.route('/predict.html', methods=['POST', 'GET'])
def predict():
    Hotel_id = int(request.form['hotel_id'])
    Property_type = int(request.form['property_type'])
    Room_type = int(request.form['room_type'])
    Accommodates = int(request.form['accommodates'])
    Cancellation_policy = int(request.form['cancellation_policy'])
    Cleaning_fee = int(request.form['cleaning_fee'])
    City = int(request.form['city'])
    Host_identity_verified = int(request.form['host_identity_verified'])
    Instant_bookable = int(request.form['instant_bookable'])
    Latitude = float(request.form['latitude'])
    Longitude = float(request.form['longitude'])  
    Number_of_reviews = int(request.form['number_of_reviews'])
    Bedrooms = int(request.form['bedrooms'])

    prediction = model.predict(pd.DataFrame([[Hotel_id, Property_type, Room_type, Accommodates, Cancellation_policy, Cleaning_fee, City, Host_identity_verified, Instant_bookable, Latitude, Longitude, Number_of_reviews, Bedrooms]]))
    prediction = np.round(prediction, 4)
    return render_template('predict.html', prediction_text=f"The predicted value is {prediction}")

if __name__ == '__main__':  
    app.run(debug=True)
