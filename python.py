import numpy as np
import pickle
import os
import pandas as pd
from flask import Flask, request, render_template

# 🔹 Initialize Flask App
app = Flask(__name__, template_folder="templates")

# 🔹 Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Load the Model and Encoder
model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pk1")

try:
    model = pickle.load(open(model_path, 'rb'))
    encoder = pickle.load(open(encoder_path, 'rb'))
    print("✅ Model and Encoder loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure 'model.pkl' and 'encoder.pk1' exist in the correct directory.")
    exit()

# 🔹 Route to Display Home Page
@app.route('/')
def home():
    return render_template('app.html')

# 🔹 Route to Handle Predictions
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # 🔍 Print raw form data for debugging
        print("\n🔍 Received Form Data:", request.form)

        # Check if any input is missing or invalid
        for key, value in request.form.items():
            if value.strip() == "":
                return f"❌ Error: Missing value for '{key}'. Please enter all values."
            try:
                float(value)  # Ensure the value is numeric
            except ValueError:
                return f"❌ Error: Invalid value for '{key}'. Please enter a numeric value."

        # Convert inputs to correct types
        try:
            holiday = int(request.form.get('holiday', 0))
            temp = float(request.form.get('temp', 0.0))
            rain = float(request.form.get('rain', 0.0))
            snow = float(request.form.get('snow', 0.0))
            weather = int(request.form.get('weather', 0))
            year = int(request.form.get('year', 2024))
            month = int(request.form.get('month', 1))
            day = int(request.form.get('day', 1))
            hours = int(request.form.get('hours', 0))
            minutes = int(request.form.get('minutes', 0))
            seconds = int(request.form.get('seconds', 0))
        except ValueError as e:
            return f"❌ Error: Invalid input format. {str(e)}"

        # 🔹 Print converted values for debugging
        print(f"✅ Converted Values: holiday={holiday}, temp={temp}, rain={rain}, snow={snow}, "
              f"weather={weather}, year={year}, month={month}, day={day}, hours={hours}, minutes={minutes}, seconds={seconds}")

        # 🔹 Convert Celsius to Kelvin if needed
        if temp < 100:  
            temp = temp + 273.15  
            print(f"🔄 Converted Temperature (Kelvin): {temp}")

        # Validate realistic temperature range in Kelvin
        if temp < 200 or temp > 350:
            return f"❌ Error: Temperature {temp}K is out of range. Please enter a value between 200K and 350K."

        # Create input array
        input_features = [holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]
        
        # 🔹 Print input features and their types for debugging
        print("🔍 Input Features:", input_features)
        print("🔍 Input Features Types:", [type(x) for x in input_features])

        # Define column names
        column_names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']

        # Convert to DataFrame
        data = pd.DataFrame([input_features], columns=column_names)

        # 🔹 Print DataFrame and its data types
        print("\n📊 DataFrame Contents:")
        print(data)
        print("\n📊 DataFrame Data Types:")
        print(data.dtypes)

        # Ensure all values are numeric
        data = data.apply(pd.to_numeric, errors='coerce')

        # Check for missing values (NaN)
        if data.isna().sum().sum() > 0:
            return "❌ Error: Input contains missing or invalid values (NaN). Please fill out all fields correctly."

        # Print encoder details
        print("\n🔍 Encoder Details:", encoder)

        # Transform Data Using Preloaded Encoder
        data_scaled = encoder.transform(data)

        # Print scaled data for debugging
        print("\n🔍 Scaled Data:")
        print(data_scaled)
        print("🔍 Scaled Data Shape:", data_scaled.shape)

        # Make Prediction
        prediction = model.predict(data_scaled)

        # Print prediction for debugging
        print("\n🔍 Prediction:", prediction)

        # Display Result in UI
        return render_template("app.html", prediction_text=f"🚗 Estimated Traffic Volume: {prediction[0]:,.0f}")

    except Exception as e:
        return f"❌ Error: {str(e)}"
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
