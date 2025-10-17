from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS  # Important for React app

app = Flask(__name__)
CORS(app)

# Load the model and encoders once at startup
model = pickle.load(open("crop_yield_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract fields
        state = data["state"]
        district = data["district"]
        season = data["season"]
        crop = data["crop"]
        acres = float(data["acres"])

        # Encode using saved encoders
        state_encoded = label_encoders["State_Name"].transform([state])[0]
        district_encoded = label_encoders["District_Name"].transform([district])[0]
        season_encoded = label_encoders["Season"].transform([season])[0]
        crop_encoded = label_encoders["Crop"].transform([crop])[0]

        # Prepare input dataframe
        new_data = pd.DataFrame(
            [[state_encoded, district_encoded, season_encoded, crop_encoded, acres]],
            columns=["State_Name", "District_Name", "Season", "Crop", "Area"],
        )

        # Predict
        prediction = model.predict(new_data)
        print(f"Input data: {data}")
        print(f"Encoded data: {new_data}")
        print(f"Prediction result: {prediction}")
        return jsonify({"yield": float(prediction[0])})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
