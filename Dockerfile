FROM python:3.10

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt

RUN pip install -r requirements.txt

COPY . /api
 
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]