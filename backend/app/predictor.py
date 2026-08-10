
import joblib

from app.preprocessing import clean_text


# Model files
model = "app/ml_models/multinomial_logistic_model.pkl"
tfidf = "app/ml_models/tfidf_vectorizer.pkl"


class SentimentPredictor:

    def __init__(self): 
       self.model = joblib.load(model) 
       self.vectorizer = joblib.load(tfidf)
    @property
    def is_ready(self):
        return self.model is not None and self.vectorizer is not None
    # Predict review
    def predict_one(self, text):
        cleaned_text = clean_text(text)

        vectorized_text = self.vectorizer.transform([cleaned_text])

        prediction = self.model.predict(vectorized_text)[0]

        probabilities = self.model.predict_proba(vectorized_text)[0]

        return {
            "review": text,
            "sentiment": prediction,
            "probabilities": {
                cls: round(float(prob), 4)
                for cls, prob in zip(self.model.classes_, probabilities)
            }
        }


predictor = SentimentPredictor()
