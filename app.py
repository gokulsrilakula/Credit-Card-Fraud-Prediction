from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        cc_num = request.form['cc_num']
        merchant = request.form['merchant']
        category = request.form['category']
        amt = float(request.form['amt'])
        gender = request.form['gender']
        city = request.form['city']
        state = request.form['state']
        zip_code = int(request.form['zip'])
        lat = float(request.form['lat'])
        long = float(request.form['long'])
        city_pop = int(request.form['city_pop'])
        job = request.form['job']
        dob = request.form['dob']
        unix_time = int(request.form['unix_time'])
        merch_lat = float(request.form['merch_lat'])
        merch_long = float(request.form['merch_long'])

        # Create a DataFrame for the new transaction
        new_transaction = pd.DataFrame({
            'cc_num': [cc_num],
            'merchant': [merchant],
            'category': [category],
            'amt': [amt],
            'gender': [gender],
            'city': [city],
            'state': [state],
            'zip': [zip_code],
            'lat': [lat],
            'long': [long],
            'city_pop': [city_pop],
            'job': [job],
            'dob': [dob],
            'unix_time': [unix_time],
            'merch_lat': [merch_lat],
            'merch_long': [merch_long]
        })

        # Preprocess the new transaction data to match training data format
        preprocessor = model.named_steps['preprocessor']
        new_transaction_preprocessed = preprocessor.transform(new_transaction)

        # Predict fraud
        prediction = model.named_steps['classifier'].predict(new_transaction_preprocessed)
        prediction_proba = model.named_steps['classifier'].predict_proba(new_transaction_preprocessed)[:, 1]

        result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
        probability = prediction_proba[0]

        return render_template('index.html', result=result, probability=probability)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)