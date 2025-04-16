FROM python:3.9-slim

WORKDIR /app

# Install the necessary libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the necessary folders
COPY app.py .
COPY train_model.py .
COPY templates/ templates/

# Create folders for models and data
RUN mkdir -p models data

# Environmental variable setting
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

# SCRIPT boot: Model training and then launch the application
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]