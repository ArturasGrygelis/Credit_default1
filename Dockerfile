FROM python:3.9-slim 
RUN apt-get update \
    && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements.txt 
RUN pip install --upgrade pip \
    && pip install -r requirements.txt  
COPY ["lgb_model_main.joblib", "app.py", "./"] .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"] 