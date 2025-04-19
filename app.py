from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Paths to model and data info
MODEL_DIR = 'models'
DATA_DIR = 'data'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')


def load_model_info():
    """Load information about the best model"""
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def get_feature_count():
    """Get the number of features from the most recent data info file"""
    if not os.path.exists(DATA_DIR):
        return 5  # Default value if data info is not found

    # Find the latest data info file
    data_info_files = [f for f in os.listdir(DATA_DIR) if f.startswith('data_info_') and f.endswith('.json')]
    if not data_info_files:
        return 5

    # Sort by time and get the latest file
    latest_file = sorted(data_info_files)[-1]

    try:
        with open(os.path.join(DATA_DIR, latest_file), 'r') as f:
            data_info = json.load(f)
            return data_info.get('n_features', 5)
    except:
        return 5


@app.route('/', methods=['GET', 'POST'])
def index():
    """Homepage of the application"""
    prediction = None
    error = None
    model_info = load_model_info()

    # Check if the best model exists
    has_model = os.path.exists(BEST_MODEL_PATH)

    # Handle prediction form
    if request.method == 'POST':
        try:
            if not has_model:
                error = "No trained model found. Please run train_model.py first."
            else:
                # Load model
                model = joblib.load(BEST_MODEL_PATH)

                # Get input from form
                raw_input = request.form.get('features', '')
                input_data = [float(x.strip()) for x in raw_input.split(',')]

                # Check feature count
                n_features = get_feature_count()
                if len(input_data) != n_features:
                    error = f"Expected {n_features} features, separated by commas."
                else:
                    # Make prediction
                    input_array = np.array(input_data).reshape(1, -1)
                    prediction = int(model.predict(input_array)[0])

                    # If model supports predict_proba, get probabilities
                    proba = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_array)[0]
                        proba = {i: float(p) for i, p in enumerate(proba)}
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           error=error,
                           model_info=model_info,
                           has_model=has_model,
                           n_features=get_feature_count())


@app.route('/predict', methods=['POST'])
def predict_api():
    """Prediction API"""
    try:
        if not os.path.exists(BEST_MODEL_PATH):
            return jsonify({'error': 'No trained model found!'}), 400

        model = joblib.load(BEST_MODEL_PATH)

        data = request.get_json()
        features = data.get('features')

        if not features or not isinstance(features, list):
            return jsonify({'error': 'Please send data as a list!'}), 400

        n_features = get_feature_count()
        if len(features) != n_features:
            return jsonify({'error': f'Expected {n_features} features!'}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(input_array)[0])

        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_array)[0].tolist()

        return jsonify({
            'prediction': prediction,
            'probabilities': proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info_api():
    """API to return best model info in UI-friendly format"""
    model_info = load_model_info()
    if model_info:
        return jsonify({
            "run_id": model_info.get('timestamp', 'N/A'),
            "model_name": model_info.get('model_name', 'N/A'),
            "parameters": model_info.get('best_params', {}),
            "metrics": {"accuracy": model_info.get('accuracy', 0.0)}
        })
    else:
        return jsonify({"error": "No model has been trained yet!"}), 404


if __name__ == '__main__':
    app.run(debug=True)
