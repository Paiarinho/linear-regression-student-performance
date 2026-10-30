import joblib
import pandas as pd
from .config import MODEL, PREPROCESSOR


class StudentScorePredictor:
    def __init__(self):
        self.model = joblib.load(MODEL)
        self.preprocessor = joblib.load(PREPROCESSOR)

    def predict(self, student_data):
        if isinstance(student_data, dict):
            df_new = pd.DataFrame([student_data])
        else:
            df_new = student_data.copy()

        X_processed = self.preprocessor.transform(df_new)
        
        prediction = self.model.predict(X_processed)[0]
        
        return round(prediction, 2)