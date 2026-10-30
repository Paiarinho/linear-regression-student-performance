from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .api_model import StudentScorePredictor


app = FastAPI(title="Enade Score Prediction API", version="1.0")

try:
    predictor = StudentScorePredictor()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    predictor = None

class StudentData(BaseModel):
    sex:str
    race_ethnicity: str = Field(alias="race/ethnicity")
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int
    class Config:
        populate_by_name=True


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is online and ready to receive data!"}

@app.post("/predict")
def predict_score(student: StudentData):
    """
    Receives student data through a JSON request.
    FastAPI automatically validates the input against the StudentData model.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded correctly.")

    try:

        student_dict = student.model_dump(by_alias=True)
        
  
        predicted_score = predictor.predict(student_dict)
        
        return {
            "message": "Prediction successful!",
            "predicted_enade_score": predicted_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")