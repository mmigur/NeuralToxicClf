from catboost import CatBoostClassifier
from fastapi import FastAPI

from model.text_preprocessing import preprocessing
from config import APP_DESCRIPTION, APP_VERSION, APP_TITLE, MODEL_PATH, OK_STATUS

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

MODEL = CatBoostClassifier().load_model(MODEL_PATH)

@app.get("/comment_clf/{comment}")
def predict(comment: str) -> dict:
    comment = preprocessing(comment)
    result_request = {
        "status": OK_STATUS,
        "label": MODEL.predict(comment)
    }
    return result_request

