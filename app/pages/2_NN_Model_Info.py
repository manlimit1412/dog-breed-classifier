"""
pages/2_NN_Model_Info.py
หน้าอธิบาย Neural Network (EfficientNetB0)
"""

import streamlit as st

st.set_page_config(page_title="🗁NN Model Info", page_icon="", layout="wide")

st.title("Neural Network — EfficientNetB0")
st.caption("จำแนกสายพันธุ์สุนัข 70 สายพันธุ์จากรูปภาพ")

# ---- Dataset ----
st.header("1. การเตรียมข้อมูล (Image Preprocessing)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dataset")
    st.markdown("""
**ที่มา:** 70 Dog Breeds Image Dataset — Kaggle  
**ประเภท:** Unstructured Data (รูปภาพ)  
**จำนวน:** ~10,000 รูป (70 สายพันธุ์ × ~140 รูป/สายพันธุ์)

**โครงสร้าง Dataset:**
```
images/
  train/
    labrador/      ← 140 รูป
    poodle/        ← 140 รูป
    ...
  test/
    labrador/      ← 35 รูป
    ...
```
    """)

with col2:
    st.subheader("ขั้นตอน Preprocessing")
    st.markdown("""
**ความไม่สมบูรณ์ของข้อมูล:**
- รูปภาพขนาดไม่เท่ากัน
- Brightness/contrast แตกต่างกัน
- บางสายพันธุ์มีรูปน้อยกว่า (class imbalance)
- มีรูปที่ถ่ายในมุมต่าง ๆ

**วิธีแก้ไข:**
1. **Resize** → ปรับขนาดทุกภาพเป็น 224×224 pixels
2. **Rescale** → หารด้วย 255 (normalize 0–1)
3. **Data Augmentation** (train only):
   - Horizontal flip
   - Rotation ±20°
   - Zoom ±20%
   - Width/Height shift ±15%
   - Brightness adjustment
4. **Validation Split** → แบ่ง 15% จาก train เป็น validation
    """)

st.divider()

# ---- Algorithm Theory ----
st.header("2. ทฤษฎีของอัลกอริทึม")

tabs = st.tabs(["CNN พื้นฐาน", "EfficientNet", "Transfer Learning", "Fine-tuning"])

with tabs[0]:
    st.subheader("Convolutional Neural Network (CNN)")
    st.markdown("""
CNN ประกอบด้วย 3 ส่วนหลัก:

**1. Convolutional Layer**  
ใช้ filter (kernel) ขนาด k×k sliding ผ่านภาพ:
$$\\text{Feature Map}_{i,j} = \\sum_{m,n} I(i+m, j+n) \\cdot K(m,n) + b$$

**2. Pooling Layer**  
ลด spatial dimensions (Max Pooling เลือกค่าสูงสุดในแต่ละ region)

**3. Fully Connected Layer**  
แปลง feature map → class probabilities ผ่าน Dense layers + Softmax:
$$P(class=c) = \\frac{e^{z_c}}{\\sum_j e^{z_j}}$$
    """)

with tabs[1]:
    st.subheader("EfficientNet Architecture")
    st.markdown("""
EfficientNet (Tan & Le, 2019) ใช้ **Compound Scaling** — scale 3 dimension พร้อมกัน:

$$d = \\alpha^\\phi, \\quad w = \\beta^\\phi, \\quad r = \\gamma^\\phi$$

โดย $d$ = depth, $w$ = width, $r$ = resolution, $\\phi$ = scaling coefficient

**EfficientNetB0** คือ baseline (B0) ที่เล็กที่สุด แต่ยัง:
- แม่นยำกว่า ResNet50 บน ImageNet
- Parameters น้อยกว่า 5 เท่า
- ใช้ **MBConv blocks** (Mobile Inverted Bottleneck + Squeeze-and-Excitation)

**โครงสร้าง head ที่เพิ่มเติม:**
```
EfficientNetB0 (frozen)
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dense(256, ReLU) + Dropout(0.4)
  → Dense(128, ReLU) + Dropout(0.3)
  → Dense(70, Softmax)
```
    """)

with tabs[2]:
    st.subheader("Transfer Learning")
    st.markdown("""
Transfer Learning นำ **weights ที่เทรนบน ImageNet** (1.2M รูป, 1000 class) มาใช้:

**ทำไมถึงได้ผล?**  
ชั้นต้น ๆ ของ CNN เรียนรู้ **general features** (edges, textures, shapes)  
ที่ใช้ได้กับทุก vision task รวมถึงการจำแนกสุนัข

**Phase 1 — Frozen Base:**
- Lock weights ของ EfficientNetB0 ทั้งหมด
- เทรนเฉพาะ classification head ที่เพิ่มเติม
- ใช้ learning rate สูง (1e-3)
- เทรน 10 epochs

**ข้อดี:** เร็ว, ไม่ destroy pretrained weights, head เรียนรู้ distribution ของ dataset ใหม่ก่อน
    """)

with tabs[3]:
    st.subheader("Fine-tuning")
    st.markdown("""
**Phase 2 — Fine-tuning:**
- Unfreeze 30 ชั้นบนสุดของ EfficientNetB0
- ใช้ learning rate ต่ำมาก (1e-5) เพื่อปรับ weights ละเอียด
- เทรนอีก 10 epochs

**Callbacks ที่ใช้:**
- `EarlyStopping(patience=5)` — หยุดถ้า val_accuracy ไม่ดีขึ้น
- `ReduceLROnPlateau(factor=0.5, patience=3)` — ลด LR เมื่อ stuck
- `ModelCheckpoint` — บันทึก best model

**Loss Function:** Categorical Cross-Entropy  
$$L = -\\sum_c y_c \\log(\\hat{y}_c)$$

**Optimizer:** Adam (Adaptive Moment Estimation)  
$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$
    """)

st.divider()

# ---- Development Steps ----
st.header("3. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
```
1. Download dataset จาก Kaggle (70 Dog Breeds)
2. สร้าง ImageDataGenerator (train + val + test)
3. Data Augmentation สำหรับ train set
4. Build model: EfficientNetB0 (frozen) + custom head
5. Compile: optimizer=Adam(1e-3), loss=categorical_crossentropy
6. Phase 1 Training: เทรน 10 epochs, base frozen
7. Plot learning curves
8. Unfreeze top 30 layers ของ base model
9. Compile: optimizer=Adam(1e-5) — fine-tuning
10. Phase 2 Training: เทรนอีก 10 epochs
11. Evaluate บน test set
12. Save model (.h5) + class_indices.json
```
""")

st.divider()

# ---- References ----
st.header("4. แหล่งอ้างอิง")
st.markdown("""
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*.
- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*.
- Yosinski, J. et al. (2014). How transferable are features in deep neural networks? *NeurIPS*.
- TensorFlow/Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras
- Dataset: https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
""")
