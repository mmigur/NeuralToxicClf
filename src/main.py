from catboost import CatBoostClassifier
from fastapi import FastAPI

from src.model.text_preprocessing import remove_punctuation, lemmatize, vectorize
from src.config import APP_DESCRIPTION, APP_VERSION, APP_TITLE, MODEL_PATH, OK_STATUS

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

MODEL = CatBoostClassifier().load_model(MODEL_PATH)

@app.get("/comment_clf/{comment}")
def predict(comment: str) -> dict:
    comment = vectorize(lemmatize(remove_punctuation(comment)))
    result_request = {
        "status": OK_STATUS,
        "label": MODEL.predict(comment)
    }
    return result_request