# train_model.py - Script huấn luyện và lưu mô hình
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# Tạo thư mục lưu trữ nếu chưa tồn tại
MODEL_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Đường dẫn đến mô hình tốt nhất
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')


def generate_data(n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
                  n_classes=2, random_state=42):
    """
    Tạo dữ liệu tổng hợp sử dụng make_classification và lưu vào file
    """
    print(f"Tạo dữ liệu với {n_samples} mẫu và {n_features} đặc trưng...")

    # Tạo dữ liệu
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tạo DataFrame để lưu
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    # Lưu dữ liệu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(DATA_DIR, f'dataset_{timestamp}.csv')
    df.to_csv(data_path, index=False)

    # Lưu thông tin về dữ liệu
    data_info = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'n_redundant': n_redundant,
        'n_classes': n_classes,
        'random_state': random_state,
        'data_path': data_path,
        'timestamp': timestamp
    }

    with open(os.path.join(DATA_DIR, f'data_info_{timestamp}.json'), 'w') as f:
        json.dump(data_info, f)

    print(f"Đã lưu dữ liệu tại: {data_path}")
    return X_train, X_test, y_train, y_test, n_features


def train_model(model_name, X_train, y_train, X_test, y_test, param_grid=None):
    """
    Huấn luyện và tinh chỉnh mô hình
    """
    print(f"Huấn luyện mô hình {model_name}...")

    # Chọn mô hình dựa trên tên
    if model_name == 'LogisticRegression':
        base_model = LogisticRegression(max_iter=10000)
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            }
    elif model_name == 'DecisionTree':
        base_model = DecisionTreeClassifier()
        if param_grid is None:
            param_grid = {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
    elif model_name == 'RandomForest':
        base_model = RandomForestClassifier()
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
    else:
        raise ValueError(f"Mô hình {model_name} không được hỗ trợ")

    # Tinh chỉnh siêu tham số
    print("Tinh chỉnh siêu tham số...")
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Lấy mô hình tốt nhất
    best_model = grid_search.best_estimator_

    # Đánh giá mô hình trên tập kiểm tra
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Tạo báo cáo
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nĐộ chính xác của {model_name}: {accuracy:.4f}")
    print(f"\nBáo cáo phân loại:\n{report}")
    print(f"\nMa trận nhầm lẫn:\n{conf_matrix}")

    # Lưu thông tin mô hình
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info = {
        'model_name': model_name,
        'best_params': grid_search.best_params_,
        'accuracy': float(accuracy),
        'timestamp': timestamp
    }

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR, f'{model_name}_{timestamp}.pkl')
    joblib.dump(best_model, model_path)

    print(f"Đã lưu mô hình tại: {model_path}")
    return best_model, accuracy, model_info, model_path


def compare_and_save_best_model(model_info):
    """
    So sánh và lưu mô hình tốt nhất
    """
    # Kiểm tra xem đã có thông tin mô hình tốt nhất chưa
    best_info = None
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                best_info = json.load(f)
        except:
            best_info = None

    # Nếu chưa có hoặc mô hình mới tốt hơn
    if best_info is None or model_info['accuracy'] > best_info['accuracy']:
        # Lưu thông tin mô hình tốt nhất
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f)

        # Lưu mô hình tốt nhất
        model_path = os.path.join(MODEL_DIR, f"{model_info['model_name']}_{model_info['timestamp']}.pkl")
        best_model = joblib.load(model_path)
        joblib.dump(best_model, BEST_MODEL_PATH)

        print(
            f"\nMô hình {model_info['model_name']} được chọn là mô hình tốt nhất với độ chính xác {model_info['accuracy']:.4f}")
        return True
    else:
        print(
            f"\nGiữ nguyên mô hình tốt nhất hiện tại ({best_info['model_name']}) với độ chính xác {best_info['accuracy']:.4f}")
        return False


def main():
    # Tạo dữ liệu
    X_train, X_test, y_train, y_test, n_features = generate_data(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2
    )

    # Các mô hình để thử nghiệm
    models = ['LogisticRegression', 'DecisionTree', 'RandomForest']

    best_model_updated = False

    # Huấn luyện và đánh giá từng mô hình
    for model_name in models:
        print(f"\n{'-' * 50}")
        model, accuracy, model_info, _ = train_model(
            model_name, X_train, y_train, X_test, y_test
        )

        # Kiểm tra và cập nhật mô hình tốt nhất
        if compare_and_save_best_model(model_info):
            best_model_updated = True

    if best_model_updated:
        print("\nĐã cập nhật mô hình tốt nhất.")
    else:
        print("\nMô hình tốt nhất không thay đổi.")


if __name__ == "__main__":
    main()