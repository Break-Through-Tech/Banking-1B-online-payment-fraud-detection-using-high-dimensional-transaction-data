import streamlit as st
from model_loader import load_model

st.title("Analyst")
model = load_model()
# TODO: table of flagged transactions with score and top SHAP reasons,
# filter by score threshold, show cardholder responses from SQLite