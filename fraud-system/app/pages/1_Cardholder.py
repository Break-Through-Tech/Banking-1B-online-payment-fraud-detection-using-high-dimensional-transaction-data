import streamlit as st
from model_loader import load_model

st.title("Cardholder confirmation")
model = load_model()
# TODO: list flagged transactions, Confirm / Not me buttons, write label to SQLite