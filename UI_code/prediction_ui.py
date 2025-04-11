import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Suppress warnings
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"
os.environ["PYDEVD_DISABLE_SLOW_RESOLVE_WARNINGS"] = "1"

# Load the model and encoders
model = joblib.load(r'pickles/naive_bayes_model.pkl')
label_encoders = joblib.load(r'pickles/label_encoders.pkl')
target_encoder = joblib.load(r'pickles/target_encoder.pkl')
model_features = joblib.load(r"pickles/model_features.pkl")

# Load dataframe for dropdown values
df_selected = pd.read_excel(r'input_dfs/df_selected.xlsx')

# Filter out only relevant features
model_only_features = [x for x in df_selected.columns if x in model_features and x != 'sales_category']

@app.route('/ui', methods=['GET', 'POST'])
def index():
    result = None
    selected_values = {}

    # Preparing dropdown values
    unique_values = {feature: sorted(df_selected[feature].dropna().unique()) for feature in model_only_features}

    if request.method == 'POST':
        input_data = {}

        for feature in model_only_features:
            value = request.form.get(feature, "")  # Get user-selected value
            selected_values[feature] = value  # Store selected value for retention

            # Apply encoding if necessary
            if feature in label_encoders:
                encoded_value = label_encoders[feature].transform([value])[0]
                input_data[feature] = encoded_value
            else:
                input_data[feature] = value

        # Convert input data to DataFrame
        X_new = pd.DataFrame([input_data], columns=model_only_features)

        # Make predictions
        prediction = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)

        # Decode the prediction
        predicted_category = target_encoder.inverse_transform(prediction)[0]
        formatted_probabilities = ", ".join(f"{p:.2%}" for p in prediction_proba[0])

        result = f'<strong>Predicted Category:</strong> {predicted_category} <br><strong>Probabilities:</strong> {formatted_probabilities}'

    return render_template(
        'index.html',
        features=model_only_features,
        unique_values=unique_values,
        selected_values=selected_values,
        result=result
    )

if __name__ == '__main__':
    app.run(debug=False)
