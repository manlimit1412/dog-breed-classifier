"""
pages/3_ML_Predict.py
หน้าทดสอบ Ensemble ML Model
"""

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="ML Predict", page_icon="", layout="wide")
st.title("ทดสอบ Ensemble ML Model")
st.caption("กรอกข้อมูลลักษณะสุนัข")

# ---- Load Model ----
@st.cache_resource
def load_model():
    model_path = "models/ensemble_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/encoders.pkl"

    if not os.path.exists(model_path):
        return None, None, None

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    return model, scaler, encoders

model, scaler, encoders = load_model()

if model is None:
    st.warning("ยังไม่พบโมเดล กรุณาเทรนโมเดลก่อนโดยรัน `train_ensemble.py`")
    st.info("สาธิต UI ในโหมด Demo (ผลลัพธ์จำลอง)")
    demo_mode = True
else:
    demo_mode = False
    st.success("โหลดโมเดลสำเร็จ")

st.divider()

# ---- Input Form ----
st.subheader("กรอกข้อมูลสุนัข")

col1, col2, col3 = st.columns(3)

with col1:
    weight_kg = st.number_input(
        "น้ำหนักเฉลี่ย (kg)", min_value=1.0, max_value=100.0, value=25.0, step=0.5
    )
    height_cm = st.number_input(
        "ความสูงเฉลี่ย (cm)", min_value=10.0, max_value=100.0, value=55.0, step=1.0
    )

with col2:
    lifespan = st.number_input(
        "อายุขัยเฉลี่ย (ปี)", min_value=5.0, max_value=20.0, value=12.0, step=0.5
    )
    intel_rank = st.slider(
        "อันดับความฉลาด (1=ฉลาดสุด)", min_value=1, max_value=138, value=50
    )

with col3:
    origin_options = [
        "germany", "united kingdom", "united states", "france",
        "japan", "australia", "canada", "ireland", "china", "other"
    ]
    origin = st.selectbox("ประเทศต้นกำเนิด", origin_options)

# Feature engineering (เหมือนกับตอน train)
weight_height_ratio = weight_kg / (height_cm + 1e-5)
if intel_rank <= 30:
    intel_tier_str = "high"
elif intel_rank <= 60:
    intel_tier_str = "medium"
else:
    intel_tier_str = "low"

st.divider()

# ---- Predict ----
if st.button("🔎︎ ทำนายสายพันธุ์", type="primary", use_container_width=True):

    if demo_mode:
        # Demo mode
        import random
        groups = ["Herding", "Hound", "Non-Sporting", "Sporting", "Terrier", "Toy", "Working"]
        probs = np.random.dirichlet(np.ones(7))
        pred_idx = np.argmax(probs)
        pred_group = groups[pred_idx]

        st.success(f"### ทำนายสายพันธุ์: **{pred_group}**")
        st.caption("(ผลลัพธ์จำลอง — Demo Mode)")

        st.subheader("ความน่าจะเป็นทุกกลุ่ม")
        for g, p in sorted(zip(groups, probs), key=lambda x: -x[1]):
            bar_color = "🟦" if g == pred_group else "⬜"
            st.markdown(f"{bar_color} **{g}** — {p*100:.1f}%")
            st.progress(float(p))

    else:
        try:
            # Encode origin
            le_origin = encoders.get("origin")
            if le_origin and origin in le_origin.classes_:
                origin_encoded = le_origin.transform([origin])[0]
            else:
                origin_encoded = 0

            intel_tier_map = {"high": 0, "medium": 1, "low": 2}
            intel_tier_encoded = intel_tier_map[intel_tier_str]

            # Build feature vector (ต้องตรงกับ columns ที่ใช้ตอน train)
            X = np.array([[
                weight_kg, height_cm, lifespan, intel_rank,
                origin_encoded, weight_height_ratio, intel_tier_encoded
            ]])
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_group = encoders["target"].inverse_transform([pred_idx])[0]
            classes = encoders["target"].classes_

            st.success(f"### กลุ่มสายพันธุ์ที่ทำนาย: **{pred_group.upper()}**")

            st.subheader("ความน่าจะเป็นทุกกลุ่ม")
            sorted_pairs = sorted(zip(classes, probs), key=lambda x: -x[1])
            for cls, prob in sorted_pairs:
                bar_color = "🟦" if cls == pred_group else "⬜"
                st.markdown(f"{bar_color} **{cls}** — {prob*100:.1f}%")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

st.divider()

# ---- Feature summary ----
with st.expander("Feature ที่ใช้ในการทำนาย"):
    st.markdown(f"""
    | Feature | ค่า |
    |---------|-----|
    | น้ำหนัก | {weight_kg} kg |
    | ความสูง | {height_cm} cm |
    | อายุขัย | {lifespan} ปี |
    | อันดับความฉลาด | {intel_rank} |
    | ประเทศต้นกำเนิด | {origin} |
    | weight/height ratio | {weight_height_ratio:.4f} |
    | intelligence tier | {intel_tier_str} |
    """)
