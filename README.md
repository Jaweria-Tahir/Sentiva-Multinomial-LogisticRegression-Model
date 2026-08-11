# Sentiva — Women's Clothing Review Sentiment Analyzer

Sentiva is an end-to-end machine learning project that classifies women's clothing e-commerce reviews as **Positive** or **Not Positive** using a **Logistic Regression** classifier trained on TF-IDF text features. It was built primarily as an **ML practice project**, covering the full pipeline: data cleaning, EDA, feature engineering, model training/evaluation, and deployment behind a FastAPI backend with a lightweight HTML/JS frontend.
---

##  Project Link
https://sentiva-multinomial-logistic-regres.vercel.app/
---


##  Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entrypoint + CORS config
│   │   ├── predictor.py         # Loads model/vectorizer, runs predictions
│   │   ├── preprocessing.py     # Text cleaning function (shared w/ training)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── ml_models/           # Saved model + vectorizer (.pkl)
│   │   └── routers/
│   │       └── predict.py       # /predict and /health endpoints
│   ├── ml_training/
│   │   ├── data/                # Raw Kaggle dataset (CSV)
│   │   └── notebook/
│   │       └── Sentiva - ModelTraining.ipynb   # Full training pipeline
│   └── requirements.txt
└── frontend/
    └── index.html                # Single-file UI (calls the API)
```

---

##  Model Training — Full Walkthrough

The model was trained in `ml_training/notebook/Sentiva - ModelTraining.ipynb`. Below is exactly what happens, step by step.

### 1. Dataset

The dataset is Kaggle's **Women's E-Commerce Clothing Reviews**:

 **Source:** [Women's E-Commerce Clothing Reviews — Kaggle (nicapotato)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

It contains ~23,000 real customer reviews of women's clothing items, including review text, star rating (1–5), recommendation flag, and product metadata (department, division, class).

### 2. Data Cleaning & EDA

- Dropped irrelevant columns: `Unnamed: 0`, `Clothing ID`, `Age`, `Title`, `Positive Feedback Count`, `Division Name`, `Class Name`.
- Checked and visualized missing values per column.
- Dropped rows with missing `Review Text` and missing `Department Name`.
- Removed duplicate rows.
- Stripped leading/trailing whitespace from text columns.
- Inspected review length distribution and rating/recommendation value ranges for outliers.
- Consolidated the `Intimate` department into `Sleep Wears` for cleaner categories.
- Explored relationships between `Rating`, `Department Name`, `Recommended IND`, and the derived sentiment label via cross-tabulations and correlation.

### 3. Target Label Creation

Since the raw data only has a 1–5 star rating, a binary sentiment label was engineerd:
- **Positive** → rating of 4 or 5
- **Not Positive** → rating of 1, 2, or 3

### 4. Text Preprocessing

A shared `clean_text()` function (duplicated identically in `preprocessing.py` for inference) is applied to every review.
This keeps only lowercase alphabetic characters and single spaces, producing clean input for vectorization.

### 5. Train/Test Split

### 6. Feature Engineering — TF-IDF Vectorization

A couple of things worth calling out here:

- I kept negation words (`not`, `no`, `never`, etc.) out of the stopword list. Normally you'd remove them, but stuff like "not soft" means the opposite of "soft" — dropping "not" would confuse the model.
- I used `ngram_range=(1,2)` so it picks up two-word phrases like "runs small" or "not soft", not just single words.
- Capped it at `max_features=10000` so the vocab doesn't get too huge.
- Only fit the vectorizer on the training data, then just transformed the test data with it — fitting on the test set too would leak information into the model.

### 7. Model — Logistic Regression

Just a plain `LogisticRegression` from sklearn trained on the TF-IDF features. A few tweaks I made:

- `C=0.5` to add a bit more regularization since there are 10k+ features and I didn't want it overfitting.
- `class_weight='balanced'` because the dataset has way more positive reviews than negative ones — without this the model would just lean toward predicting "Positive" most of the time.
- `max_iter=1000` because it wasn't converging with the default number of iterations.

### 8. Evaluation

The model was evaluated on the held-out 20% test set using:
- **Confusion matrix** (`confusion_matrix`)
- **Accuracy** (`accuracy_score`)
- **Per-class Precision** (`precision_score`, `average=None`)
- **Per-class Recall** (`recall_score`, `average=None`)
- **Full classification report** (`classification_report`) — precision/recall/F1 per class

---

## 🔌 Backend (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health/root check |
| `/health` | GET | Returns API status + whether the model loaded successfully |
| `/predict` | POST | Accepts `{ "text": "..." }`, returns sentiment + class probabilities |

**Example request:**
```json
POST /predict
{
  "text": "The fabric is soft but the zipper broke on the first day."
}
```

**Example response:**
```json
{
  "review": "The fabric is soft but the zipper broke on the first day.",
  "sentiment": "Not Positive",
  "probabilities": {
    "Not Positive": 0.7231,
    "Positive": 0.2769
  }
}
```

### Running the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

##  Frontend

A single-file HTML/CSS/JS interface (`frontend/index.html`).

---

##  Tech Stack

- **ML:** scikit-learn (`LogisticRegression`, `TfidfVectorizer`), pandas, joblib
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS
- **Notebook:** Jupyter

---

##  Dataset Citation

This project uses the **Women's E-Commerce Clothing Reviews** dataset:

> Nicapotato. *Women's E-Commerce Clothing Reviews*. Kaggle.
> https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
