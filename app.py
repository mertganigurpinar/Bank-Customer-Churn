import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Banka Müşteri Terk Tahmini", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifact(path: str = "models/churn_model.joblib"):
    return joblib.load(path)


artifact = load_artifact()
model, threshold = artifact["pipeline"], artifact["threshold"]

st.title("🏦 Banka Müşteri Terk (Churn) Tahmini")
st.caption("Random Forest tabanlı erken uyarı modeli. Müşteri profilini girin, terk riskini görün.")

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.slider("Kredi Skoru", 300, 850, 650)
        geography    = st.selectbox("Ülke", ["France", "Germany", "Spain"])
        gender       = st.selectbox("Cinsiyet", ["Female", "Male"])
        age          = st.slider("Yaş", 18, 92, 40)
        tenure       = st.slider("Müşteri Süresi (yıl)", 0, 10, 5)
    with c2:
        balance      = st.number_input("Hesap Bakiyesi", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Ürün Sayısı", [1, 2, 3, 4])
        salary       = st.number_input("Tahmini Yıllık Maaş", min_value=0.0, value=100000.0, step=1000.0)
        has_card     = st.checkbox("Kredi kartı var", value=True)
        is_active    = st.checkbox("Aktif üye", value=True)
    submitted = st.form_submit_button("Riski Hesapla", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
    }])

    proba = float(model.predict_proba(customer)[0, 1])

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Terk Olasılığı", f"{proba:.1%}")
    m2.metric("Karar Eşiği", f"{threshold:.2f}")
    st.progress(min(proba, 1.0))

    if proba >= threshold:
        st.error("🔴 **Yüksek Risk:** Müşteri temsilcisi aramalı, kişiselleştirilmiş elde tutma teklifi sunulmalı.")
    elif proba >= threshold / 2:
        st.warning("🟠 **Orta Risk:** Müşteri izleme listesine alınmalı, sadakat kampanyalarına dahil edilmeli.")
    else:
        st.success("🟢 **Düşük Risk:** Mevcut ilişki yönetimi yeterli.")

st.caption("⚠️ Eğitim/portföy amaçlı bir demodur; gerçek kredi veya müşteri kararları için kullanılmamalıdır.")