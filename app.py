import streamlit as st
import joblib

st.title("Проверка модели")

@st.cache_resource
def load_model():
    return joblib.load("application_train.joblib")

try:
    model = load_model()
    st.success("Модель загружена")
    st.write(type(model))
except Exception as e:
    st.error(f"Ошибка: {e}")
