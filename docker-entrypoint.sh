#!/bin/bash
set -e

# Kiểm tra xem đã có mô hình tốt nhất chưa
if [ ! -f "models/best_model.pkl" ]; then
    echo "Chưa có mô hình được huấn luyện. Đang huấn luyện mô hình mới..."
    python train_model.py
    echo "Hoàn tất huấn luyện mô hình."
else
    echo "Đã phát hiện mô hình đã huấn luyện."
fi

# Khởi chạy ứng dụng Flask
echo "Khởi động ứng dụng Flask..."
flask run