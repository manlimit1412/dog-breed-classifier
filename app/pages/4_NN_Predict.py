"""
pages/4_NN_Predict.py
หน้าทดสอบ EfficientNetB0 Neural Network
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(page_title="NN Predict", page_icon="📷", layout="wide")
st.title("ทดสอบ Neural Network — Dog Breed Detection")
st.caption("อัปโหลดรูปสุนัข เพื่อทำนายสายพันธุ์ด้วย EfficientNetB0")

# ---- Load Model ----
@st.cache_resource
def load_nn_model():
    model_path = "models/efficientnet_model.h5"
    class_path = "models/class_indices.json"

    if not os.path.exists(model_path):
        return None, None

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    with open(class_path, "r") as f:
        class_indices = json.load(f)

    # Invert: index → class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_nn_model()

if model is None:
    st.warning("ยังไม่พบโมเดล กรุณาเทรนโมเดลก่อนโดยรัน `train_efficientnet.py`")
    st.info("สาธิต UI ในโหมด Demo (ผลลัพธ์จำลอง)")
    demo_mode = True
else:
    demo_mode = False
    st.success("โหลดโมเดล EfficientNetB0 สำเร็จ!")

st.divider()

# ---- Upload Image ----
st.subheader("อัปโหลดรูปสุนัข")

uploaded = st.file_uploader(
    "เลือกรูปภาพสุนัข (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

TOP_K = st.slider("แสดงผลลัพธ์ Top-K สายพันธุ์", min_value=1, max_value=10, value=5)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="รูปที่อัปโหลด", use_column_width=True)

    with col2:
        if st.button("ทำนายสายพันธุ์", type="primary", use_container_width=True):
            with st.spinner("กำลังวิเคราะห์รูปภาพ..."):

                if demo_mode:
                    # Demo mode — สุ่มผลลัพธ์
                    demo_breeds = [
                        "labrador_retriever", "golden_retriever", "german_shepherd",
                        "bulldog", "poodle", "beagle", "rottweiler", "yorkshire_terrier",
                        "boxer", "dachshund"
                    ]
                    probs = np.random.dirichlet(np.ones(10) * 0.5)
                    sorted_pairs = sorted(zip(demo_breeds, probs), key=lambda x: -x[1])[:TOP_K]
                    top_breed = sorted_pairs[0][0].replace("_", " ").title()

                    st.success(f"### สายพันธุ์: **{top_breed}**")
                    st.caption("(ผลลัพธ์จำลอง — Demo Mode)")

                else:
                    # Real prediction
                    import tensorflow as tf

                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

                    preds = model.predict(img_batch, verbose=0)[0]
                    top_indices = np.argsort(preds)[::-1][:TOP_K]
                    sorted_pairs = [
                        (idx_to_class.get(i, f"class_{i}"), preds[i])
                        for i in top_indices
                    ]
                    top_breed = sorted_pairs[0][0].replace("_", " ").title()
                    st.success(f"### สายพันธุ์: **{top_breed}**")

                st.subheader(f"Top-{TOP_K} Predictions")
                for rank, (breed, prob) in enumerate(sorted_pairs, 1):
                    breed_display = breed.replace("_", " ").title()
                    bar_color = "1" if rank == 1 else ("2" if rank == 2 else "3" if rank == 3 else "▫️")
                    st.markdown(f"{bar_color} **{breed_display}** — {prob*100:.2f}%")
                    st.progress(float(prob))

st.divider()

# ---- Model Info Summary ----
with st.expander("ข้อมูลโมเดล"):
    st.markdown("""
    | Parameter | ค่า |
    |-----------|-----|
    | Architecture | EfficientNetB0 + Custom Head |
    | Input Size | 224 × 224 × 3 |
    | Output | 70 สายพันธุ์ (Softmax) |
    | Training | Phase 1: Frozen + Phase 2: Fine-tune |
    | Optimizer | Adam (lr=1e-3 → 1e-5) |
    | Loss | Categorical Cross-Entropy |
    """)

st.caption("โมเดล Neural Network ใช้ Transfer Learning จาก EfficientNetB0 (ImageNet pretrained)")
