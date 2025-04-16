FROM python:3.9-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn và các thư mục cần thiết
COPY app.py .
COPY train_model.py .
COPY templates/ templates/

# Tạo thư mục cho models và data
RUN mkdir -p models data

# Thiết lập biến môi trường
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

# Script khởi động: huấn luyện mô hình rồi khởi chạy ứng dụng
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]