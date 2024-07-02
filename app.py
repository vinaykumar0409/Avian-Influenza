from flask import Flask, request, render_template, redirect, url_for, jsonify
from markupsafe import Markup
import joblib
import numpy as np

app = Flask(__name__)



# Load the trained model
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
com_name_le = joblib.load('com_name_le.pkl')
county_le = joblib.load('county_le.pkl')
loc_le = joblib.load('loc_le.pkl')
par_spec_le = joblib.load('par_spec_le.pkl')
sci_name_le = joblib.load('sci_name_le.pkl')
state_le = joblib.load('state_le.pkl')

@app.route('/')
def index():
    """
    The `index()` function returns the rendered template for the 'index.html' file.
    :return: The `index()` function is returning the result of calling the `render_template()` function
    with the argument `'index.html'`. This typically means that the function is rendering an HTML
    template named 'index.html'.
    """
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    """
    The `predict` function processes form data, transforms categorical features, scales values, makes a
    prediction using a machine learning model, and redirects to the results page.
    :return: The `predict` function is returning a redirect to the 'results' page with the predicted
    value as a parameter if the request method is 'POST'. If the request method is not 'POST', it is
    rendering the 'prediction-page.html' template.
    """
    if request.method == 'POST':
        # Extract form data
        features = [
            str(request.form['Scientific_Name']).lower(),
            str(request.form['Common_Name']).lower(),
            int(request.form['Year']),
            int(request.form['Month']),
            int(request.form['Day']),
            str(request.form['State']).lower(),
            str(request.form['County']).lower(),
            str(request.form['Locality']).lower(),
            float(request.form['Latitude']),
            float(request.form['Longitude']),
            str(request.form['Parent_Species']).lower(),
            int(request.form['Hour']),
            int(request.form['Minute'])
        ]        

        # Convert features to a numpy array
        features_array = np.array(features)

        # Checking for valid labels 
        if features_array[0].lower() not in sci_name_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(sci_name_le.classes_)})
        if features_array[1].lower() not in com_name_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(com_name_le.classes_)})
        if features_array[5].lower() not in state_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(state_le.classes_)})
        if features_array[6].lower() not in county_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(county_le.classes_)})
        if features_array[7].lower() not in loc_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(loc_le.classes_)})
        if features_array[10].lower() not in par_spec_le.classes_ :
            return jsonify({'error': 'Unseen label detected', 'message': 'Please enter a correct label.', 'valid_labels': list(par_spec_le.classes_)})
        
        # Using Label Encoder to transform categorical features
        features_array[0] = sci_name_le.transform([features_array[0]])[0]
        features_array[1] = com_name_le.transform([features_array[1]])[0]
        features_array[5] = state_le.transform([features_array[5]])[0]
        features_array[6] = county_le.transform([features_array[6]])[0]
        features_array[7] = loc_le.transform([features_array[7]])[0]
        features_array[10] = par_spec_le.transform([features_array[10]])[0]

        # Scaling the values
        array_scaled = scaler.transform([features_array])

        # Predict the class
        prediction = model.predict(array_scaled)[0]
        
        # Redirect to the prediction page
        return redirect(url_for('results', prediction=prediction))
    
    return render_template('prediction-page.html')

@app.route('/results')
def results():
    """
    The `results` function retrieves a prediction from query parameters and renders it in a template
    called 'results.html'.
    :return: The `results()` function is returning a rendered template called 'results.html' with the
    prediction value passed as a parameter.
    """
    # Get the prediction from the query parameters
    prediction = request.args.get('prediction')

    # Making sense of the prediction
    if prediction :
        prediction = 'Your Bird might have the Bird Flu'
    else :
        prediction = 'Your Bird is safe!'

    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
