import streamlit as st
import joblib

st.title("Проверка модели")

try:
    model = joblib.load("application_train.joblib")
    st.success("Модель успешно загружена")
    st.write(type(model))
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
