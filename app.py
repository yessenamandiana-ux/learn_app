import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk App", layout="centered")

st.title("Прогноз дефолта клиента")
st.write("Введите данные клиента для оценки риска")

@st.cache_resource
def load_model():
    return joblib.load("application_train.joblib")

model = load_model()

# Достаём список признаков из pipeline
preprocessor = model.named_steps["preprocessor"]
numeric_features = list(preprocessor.transformers_[0][2])
categorical_features = list(preprocessor.transformers_[1][2])
all_features = numeric_features + categorical_features

st.subheader("📋 Данные клиента")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Доход", value=150000.0)
    credit = st.number_input("🏦 Сумма кредита", value=500000.0)
    annuity = st.number_input("📅 Аннуитет", value=20000.0)
    children = st.number_input("👶 Количество детей", value=0)
    fam_members = st.number_input("👨‍👩‍👧‍👦 Количество членов семьи", value=2.0)
    goods_price = st.number_input("🛍 Стоимость товара", value=500000.0)

with col2:
    age = st.number_input("🎂 Возраст", value=30)
    employment = st.number_input("💼 Стаж работы (лет)", value=5)
    gender = st.selectbox("Пол", ["M", "F"])
    own_car = st.selectbox("🚗 Есть автомобиль", ["Y", "N"])
    own_realty = st.selectbox("🏡 Есть недвижимость", ["Y", "N"])
    income_type = st.selectbox(
        "💵 Тип дохода",
        ["Working", "Commercial associate", "Pensioner", "State servant"]
    )

education = st.selectbox(
    "🎓 Образование",
    [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
)

family = st.selectbox(
    "👪 Семейное положение",
    [
        "Married",
        "Single / not married",
        "Civil marriage",
        "Separated",
        "Widow",
    ]
)

housing = st.selectbox(
    "🏠 Тип жилья",
    [
        "House / apartment",
        "With parents",
        "Rented apartment",
        "Municipal apartment",
        "Office apartment",
        "Co-op apartment",
    ]
)

contract_type = st.selectbox(
    "📄 Тип кредита",
    ["Cash loans", "Revolving loans"]
)

name_type_suite = st.selectbox(
    "👥 С кем проживает / сопровождающие",
    ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A", "Other_B"]
)

occupation_type = st.text_input("💼 Профессия", value="Laborers")
organization_type = st.text_input("🏢 Тип организации", value="Business Entity Type 3")

region_rating_client = st.number_input("📍 Рейтинг региона", value=2)
region_rating_client_w_city = st.number_input("🏙 Рейтинг региона с городом", value=2)

ext_source_2 = st.number_input("📊 EXT_SOURCE_2", value=0.5, min_value=0.0, max_value=1.0)

days_registration_years = st.number_input("📝 Лет с регистрации", value=10)
days_id_publish_years = st.number_input("🪪 Лет с выдачи ID", value=5)
days_last_phone_change_years = st.number_input("📱 Лет с смены телефона", value=2)

obs_30 = st.number_input("OBS_30_CNT_SOCIAL_CIRCLE", value=0.0)
def_30 = st.number_input("DEF_30_CNT_SOCIAL_CIRCLE", value=0.0)
obs_60 = st.number_input("OBS_60_CNT_SOCIAL_CIRCLE", value=0.0)
def_60 = st.number_input("DEF_60_CNT_SOCIAL_CIRCLE", value=0.0)

live_region_not_work_region = st.selectbox("LIVE_REGION_NOT_WORK_REGION", [0, 1])
reg_region_not_work_region = st.selectbox("REG_REGION_NOT_WORK_REGION", [0, 1])
live_city_not_work_city = st.selectbox("LIVE_CITY_NOT_WORK_CITY", [0, 1])
reg_city_not_work_city = st.selectbox("REG_CITY_NOT_WORK_CITY", [0, 1])
reg_city_not_live_city = st.selectbox("REG_CITY_NOT_LIVE_CITY", [0, 1])
reg_region_not_live_region = st.selectbox("REG_REGION_NOT_LIVE_REGION", [0, 1])

weekday_appr_process_start = st.selectbox(
    "📅 День подачи заявки",
    ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
)

hour_appr_process_start = st.number_input("🕒 Час подачи заявки", value=12, min_value=0, max_value=23)

st.markdown("---")

if st.button("🔍 Оценить риск"):
    try:
        # Создаём пустую строку со ВСЕМИ признаками модели
        row = {col: np.nan for col in all_features}

        # Заполняем только те поля, которые пользователь ввёл
        row["AMT_INCOME_TOTAL"] = income
        row["AMT_CREDIT"] = credit
        row["AMT_ANNUITY"] = annuity
        row["CNT_CHILDREN"] = children
        row["CNT_FAM_MEMBERS"] = fam_members
        row["AMT_GOODS_PRICE"] = goods_price

        row["DAYS_BIRTH"] = -age * 365
        row["DAYS_EMPLOYED"] = -employment * 365
        row["DAYS_REGISTRATION"] = -days_registration_years * 365
        row["DAYS_ID_PUBLISH"] = -days_id_publish_years * 365
        row["DAYS_LAST_PHONE_CHANGE"] = -days_last_phone_change_years * 365

        row["CODE_GENDER"] = gender
        row["FLAG_OWN_CAR"] = own_car
        row["FLAG_OWN_REALTY"] = own_realty
        row["NAME_INCOME_TYPE"] = income_type
        row["NAME_EDUCATION_TYPE"] = education
        row["NAME_FAMILY_STATUS"] = family
        row["NAME_HOUSING_TYPE"] = housing
        row["NAME_CONTRACT_TYPE"] = contract_type
        row["NAME_TYPE_SUITE"] = name_type_suite
        row["OCCUPATION_TYPE"] = occupation_type
        row["ORGANIZATION_TYPE"] = organization_type

        row["REGION_RATING_CLIENT"] = region_rating_client
        row["REGION_RATING_CLIENT_W_CITY"] = region_rating_client_w_city
        row["EXT_SOURCE_2"] = ext_source_2

        row["OBS_30_CNT_SOCIAL_CIRCLE"] = obs_30
        row["DEF_30_CNT_SOCIAL_CIRCLE"] = def_30
        row["OBS_60_CNT_SOCIAL_CIRCLE"] = obs_60
        row["DEF_60_CNT_SOCIAL_CIRCLE"] = def_60

        row["LIVE_REGION_NOT_WORK_REGION"] = live_region_not_work_region
        row["REG_REGION_NOT_WORK_REGION"] = reg_region_not_work_region
        row["LIVE_CITY_NOT_WORK_CITY"] = live_city_not_work_city
        row["REG_CITY_NOT_WORK_CITY"] = reg_city_not_work_city
        row["REG_CITY_NOT_LIVE_CITY"] = reg_city_not_live_city
        row["REG_REGION_NOT_LIVE_REGION"] = reg_region_not_live_region

        row["WEEKDAY_APPR_PROCESS_START"] = weekday_appr_process_start
        row["HOUR_APPR_PROCESS_START"] = hour_appr_process_start

        input_data = pd.DataFrame([row], columns=all_features)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Результат")

        if prediction[0] == 1:
            st.error("❌ Высокий риск дефолта")
        else:
            st.success("✅ Низкий риск дефолта")

        st.metric("Вероятность дефолта", f"{probability:.2%}")

        with st.expander("Показать данные, переданные в модель"):
            st.dataframe(input_data.T)

    except Exception as e:
        st.error(f"Ошибка: {e}")
