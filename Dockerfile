FROM python:3.10

RUN mkdir /neural_toxic_api

WORKDIR /neural_toxic_api

COPY requirements.txt .

RUN  pip install -r requirements.txt

COPY . .

CMD gunicorn src.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000