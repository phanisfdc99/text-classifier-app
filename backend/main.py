from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Shipping Classification API")

shipping_labels = [
    "Booking",
    "Shipment Delay",
    "Documentation",
    "Billing",
    "Tracking",
    "Customs",
    "Container",
    "Delivery",
    "Other",
]

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Shipping Classification API is running. POST to /classify"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/classify")
def classify(input: TextInput):
    result = classifier(
        input.text,
        candidate_labels=shipping_labels,
        multi_label=False
    )
    return {
        "text": input.text,
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4),
        "all_labels": [
            {"label": label, "score": round(score, 4)}
            for label, score in zip(result["labels"], result["scores"])
        ],
    }