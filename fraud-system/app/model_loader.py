from pathlib import Path
import joblib
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "artifacts"

@st.cache_resource
def load_model():
	path = MODEL_DIR / "model.pkl"
	if not path.exists():
		return None  # dummy mode until P4 delivers
	return joblib.load(path)