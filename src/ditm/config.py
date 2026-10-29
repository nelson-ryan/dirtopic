from pathlib import Path
import logging
from datetime import datetime
import multiprocessing
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent / ".env")
DB_URL = os.environ["DB_URL"]
CSV_PATH = Path(os.environ["CSV_PATH"])

ROOT = Path(__file__).resolve().parents[2]

# NLP pipeline
SPACY_MODEL     = "en_core_web_sm"
MODELDIR = ROOT / "models" # Holds SentenceTransformer model, not BERTopic
EMBEDDING_MODEL = MODELDIR / "all-MiniLM-L6-v2"

# logging
LOGDIR = ROOT / "logs"
LOGFORMAT = "%(asctime)s | %(levelname)s | %(name)s : %(message)s"
LOGLEVEL = logging.INFO

OUTPUTDIR = ROOT / "output"

# plotting
VIZDIR = ROOT / "plots"
TMPDIR = ROOT / "tmp"

# leftovers; may or may not remain unused
OUTPUT_EXCEL    = "email_topic_insights.xlsx"
OUTPUT_HEATMAP  = "topic_heatmap.png"
OUTPUT_WORDCLOUD_DIR = "wordclouds/"  # one PNG per topic saved here
MIN_TOPIC_SIZE  = 3     # lower = more granular topics
TOP_N_TOPICS    = 15    # topics shown in heatmap
TIME_BUCKET     = "M"   # "W" = weekly, "M" = monthly
