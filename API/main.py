
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import pymorphy3

app = FastAPI(title="Юридический классификатор заявок")

# Загрузка моделей и векторизатора
priority_model = joblib.load("model_priority.pkl")
category_model = joblib.load("model_category.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Лемматизация
morph = pymorphy3.MorphAnalyzer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)

# Входной формат
class Request(BaseModel):
    text: str

# Классы в человекопонятном виде
CATEGORY_LABELS = {
    1: "договорное право",
    2: "трудовое право",
    3: "банкротство"
}

PRIORITY_LABELS = {
    1: "низкий",
    2: "средний",
    3: "высокий"
}

@app.post("/predict")
def predict(data: Request):
    # Предобработка
    processed = preprocess(data.text)
    vector = vectorizer.transform([processed])
    
    # Предсказания
    category_pred = category_model.predict(vector)[0]
    priority_pred = priority_model.predict(vector)[0]

    # Человекочитаемый вывод
    return {
        "category": CATEGORY_LABELS.get(int(category_pred), "неизвестно"),
        "priority": PRIORITY_LABELS.get(int(priority_pred), "неизвестно")
    }
