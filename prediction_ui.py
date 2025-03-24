import joblib

# # Save the model
# joblib.dump(gnb, 'naive_bayes_model.pkl')

# # Save the encoders
# joblib.dump(label_encoders, 'label_encoders.pkl')
# joblib.dump(target_encoder, 'target_encoder.pkl')


### Web Application using Flask
import os

import os
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"  # Increase timeout
os.environ["PYDEVD_DISABLE_SLOW_RESOLVE_WARNINGS"] = "1"

from flask import Flask, render_template, request
import pandas as pd
# from sklearn.externals import joblib

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('naive_bayes_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
import time
# time.sleep(20)

# Load the dataframe for unique values in UI
df_selected = pd.read_excel('prediction_uiData.xlsx')  # Replace with your actual CSV file

# Load feature names
features = joblib.load('model_features.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():  

    import time
    # time.sleep(20)
    # print(df_selected.dtypes)
    # print(df_selected.columns)
    categorical_features = df_selected.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [item for item in categorical_features if item in features]
    if "sales_category" in categorical_features:
        categorical_features.remove('sales_category')  # Ensure not to include the target

    # input_data = {}
    input_data = {}  # Initialize all features
    
    
    if request.method == 'POST':
        
        
        for feature in categorical_features:
            
            if feature in request.form:
                value = request.form[feature]
                # Use the label encoder
                encoded_value = label_encoders[feature].transform([value])[0]
                input_data[feature] = encoded_value

        X_new = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)

        # Decode the prediction
        result = target_encoder.inverse_transform(prediction)[0]
        return f'Result: {result} with probabilities {prediction_proba}'

    # Preparing values for the dropdown
    unique_values = {feature: df_selected[feature].unique() for feature in categorical_features}
    return render_template('index.html', features=categorical_features, unique_values=unique_values)

if __name__ == '__main__':
    app.run(debug=False)