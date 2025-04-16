# app.py - Ứng dụng Flask
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Đường dẫn đến mô hình và thông tin
MODEL_DIR = 'models'
DATA_DIR = 'data'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')


def load_model_info():
    """Tải thông tin về mô hình tốt nhất"""
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def get_feature_count():
    """Lấy số lượng đặc trưng từ file thông tin dữ liệu gần nhất"""
    if not os.path.exists(DATA_DIR):
        return 5  # Giá trị mặc định nếu không tìm thấy thông tin

    # Tìm file thông tin dữ liệu gần nhất
    data_info_files = [f for f in os.listdir(DATA_DIR) if f.startswith('data_info_') and f.endswith('.json')]
    if not data_info_files:
        return 5

    # Sắp xếp theo thời gian và lấy file mới nhất
    latest_file = sorted(data_info_files)[-1]

    try:
        with open(os.path.join(DATA_DIR, latest_file), 'r') as f:
            data_info = json.load(f)
            return data_info.get('n_features', 5)
    except:
        return 5


@app.route('/', methods=['GET', 'POST'])
def index():
    """Trang chủ ứng dụng"""
    prediction = None
    error = None
    model_info = load_model_info()

    # Kiểm tra xem đã có mô hình tốt nhất chưa
    has_model = os.path.exists(BEST_MODEL_PATH)

    # Xử lý form dự đoán
    if request.method == 'POST':
        try:
            # Kiểm tra xem đã có mô hình tốt nhất chưa
            if not has_model:
                error = "Chưa có mô hình được huấn luyện. Vui lòng chạy script train_model.py trước."
            else:
                # Tải mô hình
                model = joblib.load(BEST_MODEL_PATH)

                # Lấy dữ liệu từ form
                raw_input = request.form.get('features', '')
                input_data = [float(x.strip()) for x in raw_input.split(',')]

                # Kiểm tra số lượng đặc trưng
                n_features = get_feature_count()
                if len(input_data) != n_features:
                    error = f"Cần nhập đúng {n_features} đặc trưng, cách nhau bởi dấu phẩy"
                else:
                    # Dự đoán
                    input_array = np.array(input_data).reshape(1, -1)
                    prediction = int(model.predict(input_array)[0])

                    # Nếu mô hình hỗ trợ predict_proba, lấy xác suất
                    proba = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_array)[0]
                        proba = {i: float(p) for i, p in enumerate(proba)}
        except Exception as e:
            error = f"Lỗi: {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           error=error,
                           model_info=model_info,
                           has_model=has_model,
                           n_features=get_feature_count())


@app.route('/predict', methods=['POST'])
def predict_api():
    """API dự đoán"""
    try:
        # Kiểm tra xem đã có mô hình tốt nhất chưa
        if not os.path.exists(BEST_MODEL_PATH):
            return jsonify({'error': 'Chưa có mô hình được huấn luyện!'}), 400

        # Tải mô hình
        model = joblib.load(BEST_MODEL_PATH)

        # Lấy dữ liệu
        data = request.get_json()
        features = data.get('features')

        # Kiểm tra dữ liệu
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Vui lòng gửi dữ liệu dạng list!'}), 400

        # Kiểm tra số lượng đặc trưng
        n_features = get_feature_count()
        if len(features) != n_features:
            return jsonify({'error': f'Cần nhập đúng {n_features} đặc trưng!'}), 400

        # Dự đoán
        input_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(input_array)[0])

        # Nếu mô hình hỗ trợ predict_proba, lấy xác suất
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_array)[0].tolist()

        return jsonify({
            'prediction': prediction,
            'probabilities': proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)