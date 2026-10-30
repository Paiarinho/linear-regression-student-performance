from pathlib import Path

# Define the absolute path to the project's root directory
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Define the mains directories
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

# Define paths for specific dataset files
RAW_DATA_PATH = DATA_DIR / "StudentsPerformance.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_database.parquet"

# Define paths for model
MODEL = MODELS_DIR / "linear_regression_model.pkl"
PREPROCESSOR = MODELS_DIR / 'preprocessor.pkl'

