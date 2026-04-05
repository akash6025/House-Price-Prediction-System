from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open('../model/house_price_model.pkl', 'rb'))

# Load feature names
feature_names = pickle.load(open('../model/features.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = request.form.to_dict()

        # Create dictionary with correct feature names
        data = {}
        for feature in feature_names:
            # Take input if available, else default 0
            data[feature] = float(input_data.get(feature, 0))

        # Convert to DataFrame (VERY IMPORTANT)
        final_df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(final_df)

        # Convert log prediction back to normal price
        output = np.exp(prediction[0])

        return render_template(
            'index.html',
            prediction_text=f'Predicted Price: ${output:,.2f}'
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template(
            'index.html',
            prediction_text="Error in input"
        )


if __name__ == "__main__":
    app.run(debug=True)