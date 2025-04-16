# MSE_ĐM501: Hệ thống Phân loại ML với Flask

## Giới thiệu

Dự án này là một ứng dụng web Flask để thực hiện phân loại sử dụng các mô hình học máy. Ứng dụng bao gồm các tính năng:

- Tạo dữ liệu tổng hợp sử dụng `make_classification` của scikit-learn
- Huấn luyện và tinh chỉnh nhiều mô hình phân loại khác nhau
- So sánh và lưu mô hình tốt nhất
- Giao diện web để sử dụng và kiểm tra mô hình
- API để tích hợp với các hệ thống khác

## Cài đặt và Sử dụng

### Sử dụng Docker

```bash
# Kéo image từ Docker Hub
docker pull your-dockerhub-username/mse_dm501:latest

# Chạy container
docker run -d -p 5000:5000 --name mse_app your-dockerhub-username/mse_dm501:latest

# Truy cập ứng dụng tại http://localhost:5000
```

### Cài đặt trực tiếp

1. Clone repository:
```bash
git clone https://gitlab.com/your-username/MSE_ĐM501.git
cd MSE_ĐM501
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Huấn luyện mô hình:
```bash
python train_model.py
```

4. Khởi động ứng dụng Flask:
```bash
flask run
```

5. Truy cập ứng dụng tại http://localhost:5000

## Cấu trúc dự án

- `app.py`: Ứng dụng web Flask
- `train_model.py`: Script huấn luyện và lưu mô hình
- `templates/`: Thư mục chứa các file HTML 
- `models/`: Thư mục lưu các mô hình đã huấn luyện
- `data/`: Thư mục lưu dữ liệu tổng hợp và thông tin
- `Dockerfile`: File để build Docker image
- `.gitlab-ci.yml`: Cấu hình CI/CD cho GitLab

## API

### Dự đoán

**Endpoint**: `/predict`  
**Method**: POST  
**Content-Type**: application/json

**Request Body**:
```json
{
    "features": [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

**Response**:
```json
{
    "prediction": 0,
    "probabilities": [0.8, 0.2]
}
```

## Phát triển

Dự án sử dụng GitLab CI/CD để tự động hóa quy trình phát triển:

1. Mỗi commit vào nhánh main sẽ kích hoạt quy trình CI/CD
2. Kiểm tra cú pháp và các lỗi cơ bản
3. Build Docker image
4. Đẩy image lên Docker Hub

