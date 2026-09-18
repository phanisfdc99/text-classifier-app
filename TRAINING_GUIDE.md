# Training Guide: Logistics Email Classification

This guide explains how to replace the current zero-shot classifier with a supervised TF-IDF + Logistic Regression model trained on labeled logistics emails.

## Current implementation

The application currently loads Hugging Face's `facebook/bart-large-mnli` model in `backend/main.py`. That model performs zero-shot classification and is not trained on your logistics email data.

The new implementation will:

1. Read labeled emails from a CSV file.
2. Convert email text into TF-IDF features.
3. Train a Logistic Regression classifier.
4. Save the trained model as `email_classifier.joblib`.
5. Load the model from the FastAPI backend.
6. Return the predicted category and confidence scores.

## Project structure

```text
text-classifier-app/
├── backend/
│   ├── data/
│   │   └── emails.csv
│   ├── model/
│   │   └── email_classifier.joblib
│   ├── train.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── ui/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## 1. Prepare the dataset

Create `backend/data/emails.csv` with exactly two required columns: `text` and `label`.

```csv
text,label
"Please confirm the booking for container MSKU1234567","Booking"
"Can you advise when the shipment will arrive?","Tracking"
"The vessel arrival has been delayed by five days","Shipment Delay"
"Please find the customs declaration attached","Customs"
"We need the commercial invoice for this shipment","Documentation"
"Please send the outstanding freight invoice","Billing"
"The container is ready for pickup","Container"
"Please confirm the delivery address","Delivery"
"I need help with this logistics request","Other"
```

Use these labels consistently:

```text
Booking
Shipment Delay
Documentation
Billing
Tracking
Customs
Container
Delivery
Other
```

For a useful first model, use at least 50–100 examples per category. Production models normally need substantially more data. Remove personal, confidential, and sensitive information before storing or committing any dataset.

## 2. Replace backend requirements

Replace `backend/requirements.txt` with:

```text
fastapi==0.111.*
uvicorn[standard]==0.30.*
pandas==2.2.*
scikit-learn==1.5.*
joblib==1.4.*
```

The `transformers` and `torch` packages are not required for this Logistic Regression implementation.

## 3. Add the training script

Create `backend/train.py`:

```python
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = os.environ.get("DATA_PATH", "data/emails.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/email_classifier.joblib")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    if len(df) < 10:
        raise ValueError("The dataset must contain at least 10 valid rows.")
    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two different labels.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Every label must contain at least two examples.")

    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_dataset(DATA_PATH)
    print(f"Examples: {len(df)}")
    print(df["label"].value_counts().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = sorted(df["label"].unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

## 4. Train the model

From the repository root:

```bash
cd backend
python -m venv .venv
```

Activate the environment.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

Install dependencies and train:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

The generated file is:

```text
backend/model/email_classifier.joblib
```

## 5. Replace the API implementation

Replace `backend/main.py` with:

```python
import os
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = os.environ.get("MODEL_PATH", "model/email_classifier.joblib")

app = FastAPI(title="Shipping Classification API")

try:
    classifier = joblib.load(MODEL_PATH)
except FileNotFoundError as error:
    raise RuntimeError(
        f"Trained model was not found at '{MODEL_PATH}'. "
        "Run 'python train.py' before starting the API."
    ) from error


class TextInput(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Shipping Classification API is running. POST to /classify"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "model": os.path.basename(MODEL_PATH)}


@app.post("/classify")
def classify(input: TextInput) -> dict[str, Any]:
    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="The text field cannot be empty.")

    probabilities = classifier.predict_proba([text])[0]
    labels = classifier.classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)

    return {
        "text": text,
        "label": ranked[0][0],
        "score": round(float(ranked[0][1]), 4),
        "all_labels": [
            {"label": label, "score": round(float(score), 4)}
            for label, score in ranked
        ],
    }
```

The existing Streamlit UI already reads `label` and `score`, so no UI change is required.

## 6. Update the backend Dockerfile

Replace `backend/Dockerfile` with:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/email_classifier.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Train the model before building the image:

```bash
cd backend
python train.py
cd ..
docker compose build backend
```

## 7. Run and test locally

Start the API:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

Test classification:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Please advise why the container shipment is delayed"}'
```

API documentation is available at `http://localhost:8000/docs`.

## 8. Run with Docker Compose

From the repository root:

```bash
docker compose build
docker compose up
```

Open the Streamlit interface at:

```text
http://localhost:8501
```

The API is available at:

```text
http://localhost:8000
```

Stop the services:

```bash
docker compose down
```

## 9. Add `.gitignore` rules

Add these entries to `.gitignore`:

```gitignore
backend/.venv/
backend/__pycache__/
backend/model/*.joblib
backend/data/*.csv
!backend/data/example_emails.csv
```

Do not commit private datasets. Decide separately whether the trained model should be committed or stored in an artifact/model registry.

## 10. Evaluate the model

Review the classification report, not only accuracy. Pay attention to precision, recall, and F1 score for every category.

- Low recall for `Shipment Delay` means delayed shipments are being missed.
- Low precision for `Billing` means unrelated emails are being labeled as billing.
- A high score from a very small test set may not generalize.

Avoid data leakage by removing duplicates and keeping near-identical emails out of both training and test sets.

## 11. Optional confidence threshold

For production, route uncertain predictions to `Other` or manual review. For example:

```python
CONFIDENCE_THRESHOLD = 0.60
```

Use a validation set to choose the threshold rather than relying on `0.60` without testing.

## 12. Suggested commits

```bash
git add backend/data/emails.csv backend/train.py backend/requirements.txt
git commit -m "Add supervised email classifier training"

git add backend/main.py
git commit -m "Load trained classifier in API"

git add backend/Dockerfile .gitignore
git commit -m "Package trained classifier for deployment"
```

## Summary

The change replaces runtime zero-shot classification with a domain-specific supervised classifier. The model will learn from your labeled logistics emails, and the trained artifact will be loaded by FastAPI for classification.
