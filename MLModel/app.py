from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import base64
import io

import google.generativeai as genai
from PIL import Image

# --- PASTE YOUR GOOGLE AI API KEY HERE ---
GOOGLE_AI_API_KEY = 'AIzaSyCPa5hQ7k8n76ZkDhO8QPQLwIC8mp3e_kE' # <-- MAKE SURE THIS IS NOT THE PLACEHOLDER
# -------------------------------------------

# --- FIX: Add a check for the API key ---
if GOOGLE_AI_API_KEY != 'AIzaSyCPa5hQ7k8n76ZkDhO8QPQLwIC8mp3e_kE':
    print("="*50)
    print("WARNING: GOOGLE_AI_API_KEY is not set in app.py!")
    print("Soil image classification will fail.")
    print("="*50)
    
try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI. Is your API key valid? Error: {e}")

# --- 1. Load & Train Crop Model (Unchanged) ---
data = pd.read_csv('Crop_recommendation.csv')
features = ['temperature', 'humidity', 'ph', 'rainfall']
X = data[features]
y = data['label']
print("Training Crop Model with features:", features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train.values, y_train)
joblib.dump(crop_model, 'crop_recommendation_model_4_features.pkl')
crop_model = joblib.load('crop_recommendation_model_4_features.pkl')
print(f"Crop Model Accuracy (on 4 features): {accuracy_score(y_test, crop_model.predict(X_test.values))}")


# --- 2. Updated Function: Get Soil Name from Image ---
def get_soil_name_from_image(image_data_base64):
    if not image_data_base64:
        print("No Base64 image data received.")
        return None
    
    try:
        image_bytes = base64.b64decode(image_data_base64)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        print("Image decoded, sending to Google AI...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_parts = [
            "You are a soil expert. Analyze this image and identify the type of soil.",
            "Respond *only* with the common name of the soil (e.g., 'Alluvial Soil', 'Black Soil', 'Red Soil').",
            "If you are unsure, respond with 'Unknown Soil'.",
            "Image: ", img
        ]

        response = model.generate_content(prompt_parts)
        soil_name = response.text.strip()
        print(f"Google AI Response: {soil_name}")
        
        if "Soil" not in soil_name and len(soil_name.split()) > 3:
            return "Unknown Soil (from AI)"
            
        return soil_name

    except Exception as e:
        # --- FIX: Return a very clear error ---
        print(f"Error in Google AI vision call: {e}")
        # This error will now be sent back to the user
        raise Exception(f"Google AI Error: {e}")


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("ML Server Received data keys:", list(data.keys()))

        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']
        soil_name_from_text = data.get('soilName')
        image_base64 = data.get('soilImageBase64') # This will be None if not sent

        # --- 2. Determine the Soil Name (Updated Logic) ---
        determined_soil_name = "N/A" # Default
        
        if soil_name_from_text:
            determined_soil_name = soil_name_from_text
            print(f"Using soil name from text input: {determined_soil_name}")
        elif image_base64:
            print("Text name not found, trying image classification...")
            determined_soil_name = get_soil_name_from_image(image_base64)
        else:
            # --- FIX: Be specific about *why* it's N/A ---
            determined_soil_name = "N/A (No Soil Info)"
            print("No soil name or image provided.")

        # --- 3. Make Crop Prediction ---
        features_list = [temperature, humidity, ph, rainfall]
        features_arr = np.array(features_list).reshape(1, -1)
        
        prediction = crop_model.predict(features_arr)
        probabilities = crop_model.predict_proba(features_arr)
        confidence = np.max(probabilities) * 100
        
        # --- 4. Create the final response ---
        response = {
            'crop': prediction[0],
            'confidence': round(confidence, 2),
            'determinedSoilName': determined_soil_name 
        }
        
        print("ML Server Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in ML server /predict route: {e}")
        # --- FIX: Send the error as JSON ---
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render, or default to 8000 for local testing
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Python ML Server on port {port}")
    # host='0.0.0.0' is crucial for cloud access
    app.run(host='0.0.0.0', port=port)