import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("golden-retriever_4600293.png", width=80)
with col2:
    st.title("Dog Breed Classifier")
st.markdown("""
ยินดีต้อนรับสู่ระบบแบ่งประเภทสุนัข

เลือกหน้าจาก **sidebar** ด้านซ้ายเพื่อเริ่มใช้งาน:

| หน้า | รายละเอียด |
|------|------------|
| ML Model Info | อธิบายโมเดล Ensemble ML |
| NN Model Info | อธิบายโมเดล Neural Network |
| ML Predict | ทดสอบโมเดล Ensemble ML |
| NN Predict | ทดสอบโมเดล Neural Network |

---

### ที่มาของ Dataset
- **Dataset 1 (Structured):** Dog Breeds Stats — [Kaggle](https://www.kaggle.com/datasets/marshuu/dog-breeds)
- **Dataset 2 (Images):** 70 Dog Breeds Image Dataset — [Kaggle](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)
""")
