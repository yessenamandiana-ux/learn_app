import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk App", layout="centered")

st.title("Прогноз дефолта клиента")
st.write("Введите данные клиента для оценки риска")

# загрузка модели
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_10_fields.joblib")

model = load_model()

st.subheader("📋 Данные клиента")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Доход", value=150000.0)
    credit = st.number_input("🏦 Сумма кредита", value=500000.0)
    annuity = st.number_input("📅 Аннуитет", value=20000.0)
    children = st.number_input("👶 Количество детей", value=0)

with col2:
    age = st.number_input("🎂 Возраст", value=30)
    employment = st.number_input("💼 Стаж работы (лет)", value=5)
    gender = st.selectbox("Пол", ["M", "F"])

education = st.selectbox(
    "🎓 Образование",
    ["Secondary / secondary special", "Higher education", "Incomplete higher"]
)

family = st.selectbox(
    "👪 Семейное положение",
    ["Married", "Single / not married", "Civil marriage"]
)

housing = st.selectbox(
    "🏠 Тип жилья",
    ["House / apartment", "With parents", "Rented apartment"]
)

# формируем вход
input_data = pd.DataFrame([{
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "CNT_CHILDREN": children,
    "DAYS_BIRTH": -age * 365,
    "DAYS_EMPLOYED": -employment * 365,
    "CODE_GENDER": gender,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family,
    "NAME_HOUSING_TYPE": housing
}])

st.markdown("---")

if st.button("🔍 Оценить риск"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Результат")

        if prediction[0] == 1:
            st.error("❌ Высокий риск дефолта")
        else:
            st.success("✅ Низкий риск дефолта")

        st.metric("Вероятность дефолта", f"{probability:.2%}")

    except Exception as e:
        st.error(f"Ошибка: {e}")
