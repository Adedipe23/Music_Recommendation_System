from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DEMO_DATA_PATH = DATA_DIR / "demo_listening_history.csv"
MODEL_PATH = MODEL_DIR / "hybrid_recommender.joblib"
EVALUATION_REPORT_PATH = REPORT_DIR / "evaluation_results.md"

SAMPLE_USERS = 30_000
MIN_ARTIST_PLAY_COUNT = 1
MIN_USER_INTERACTIONS = 2
MAX_ARTISTS = 20_000
DEFAULT_TOP_K = 10

COLLABORATIVE_WEIGHT = 0.6
CONTENT_WEIGHT = 0.3
POPULARITY_WEIGHT = 0.1

LASTFM_API_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
RANDOM_STATE = 42

