import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA

MODEL_PATH = Path("model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)



