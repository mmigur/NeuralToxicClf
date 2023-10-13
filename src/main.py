from model.text_preprocessing import preprocessing
from catboost import CatBoostClassifier
from fastapi import FastAPI

app = FastAPI(
    title="Neural toxic clf",
    version="0.0.1",
    description="Приложение для фильтрации токсичных комментариев"
)

MODEL = CatBoostClassifier().load_model("./model/model")

@app.get("/comment_clf/{comment}")
def predict(comment: str) -> dict:
    comment = preprocessing(comment)
    result_request = {
        "status": 200,
        "label": MODEL.predict(comment)
    }
    return result_request

