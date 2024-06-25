import os
from flask import Flask, jsonify, send_from_directory, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        # Convert form data to DataFrame
        input_data = pd.DataFrame([form_data])
        # Convert all values to numeric
        input_data = input_data.apply(pd.to_numeric)

        # Assuming 'model' is loaded and 'nueva_prediccion' function exists
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
