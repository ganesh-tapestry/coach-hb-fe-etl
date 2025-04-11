import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Ensure slower resolve warnings do not appear
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"
os.environ["PYDEVD_DISABLE_SLOW_RESOLVE_WARNINGS"] = "1"

# Load the model and encoders
model = joblib.load(r'pickles\naive_bayes_model.pkl')
label_encoders = joblib.load(r'pickles\label_encoders.pkl')
target_encoder = joblib.load(r'pickles\target_encoder.pkl')
model_features = joblib.load(r"pickles\model_features.pkl")

print("len model_features",len(model_features))
# Load the dataframe for unique values in UI
df_selected = pd.read_excel(r'input_dfs\df_selected.xlsx')

# Prepare features list excluding the target
model_only_features = df_selected.columns.tolist()
model_only_features = [x for x in model_only_features if x in model_features]
if "sales_category" in model_only_features:
    model_only_features.remove('sales_category')
print("len model_only_features",len(model_only_features))

# 'bag_height', 'bag_width', 'gusset_width', 'closure_type',
#        'adjustable_strap', 'opener_design', 'top_opening_width',
#        'shoulder_option', 'logo_size', 'logo_style', 'finish_type',
#        'department_desc', 'subcollection', 'material', 'material_type_y'

@app.route('/ui', methods=['GET'])
def index():
    # Preparing values for the dropdown
    unique_values = {feature: df_selected[feature].unique() for feature in model_features}
    return render_template('index.html', features=model_only_features, unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}

    for feature in model_only_features:
        value = request.form[feature]
        if feature in label_encoders:
            # Use the label encoder
            encoded_value = label_encoders[feature].transform([value])[0]
            input_data[feature] = encoded_value
        else :
             input_data[feature] = value
             
    X_new = pd.DataFrame([input_data], columns=model_only_features)
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)

    # Decode the prediction
    result = target_encoder.inverse_transform(prediction)[0]
    return f'Result: {result} \n with probabilities: {prediction_proba}'

if __name__ == '__main__':
    app.run(debug=False)