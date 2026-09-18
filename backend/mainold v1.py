from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Text Classification API")

# Downloads automatically from Hugging Face on first run.
# Default model: distilbert-base-uncased-finetuned-sst-2-english (sentiment: POSITIVE/NEGATIVE)
classifier = pipeline("text-classification")


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Text Classification API is running. POST to /classify"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/classify")
def classify(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "text": input.text,
        "label": result["label"],
        "score": round(result["score"], 4),
    }
