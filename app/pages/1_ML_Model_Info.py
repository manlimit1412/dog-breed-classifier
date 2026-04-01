"""
pages/1_ML_Model_Info.py
หน้าอธิบาย Ensemble ML Model
"""

import streamlit as st

st.set_page_config(page_title="ML Model Info", page_icon="📊", layout="wide")

st.title("Ensemble Machine Learning Model")
st.caption("จำแนกกลุ่มพันธุ์สุนัข (Dog Breed Group Classification)")

# ---- Dataset ----
st.header("1. การเตรียมข้อมูล (Data Preprocessing)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dataset")
    st.markdown("""
**ที่มา:** Dog Breeds Stats — Kaggle  
**ประเภท:** Structured Data (CSV)  
**จำนวน:** ~350 สายพันธุ์

**Features ที่ใช้:**
- `weight_kg` — น้ำหนักเฉลี่ย (kg)
- `height_cm` — ความสูงเฉลี่ย (cm)  
- `lifespan_years` — อายุขัยเฉลี่ย (ปี)
- `intelligence_rank` — อันดับความฉลาด
- `origin` — ประเทศต้นกำเนิด (encoded)
- `weight_height_ratio` — feature ที่สร้างเพิ่ม
- `intelligence_tier` — high/medium/low (feature ที่สร้างเพิ่ม)

**Target:** `group` (Herding, Toy, Working, Sporting, ฯลฯ)
    """)

with col2:
    st.subheader("ขั้นตอน Preprocessing")
    st.markdown("""
**ความไม่สมบูรณ์ของข้อมูล:**
- Missing values ใน numeric columns
- Missing values ใน categorical columns
- Text ที่ยังไม่ได้ standardize
- Scale ของ features แตกต่างกันมาก

**วิธีแก้ไข:**
1. **Drop duplicates** — ลบแถวซ้ำ
2. **Impute numeric** — เติม missing ด้วย median
3. **Impute categorical** — เติม missing ด้วย mode
4. **Normalize text** — strip + lowercase
5. **Feature engineering** — สร้าง weight_height_ratio, intelligence_tier
6. **Label Encoding** — แปลง categorical → numeric
7. **Standard Scaling** — Z-score normalization
8. **Train/Test Split** — 80:20 พร้อม stratify
    """)

st.divider()

# ---- Algorithm Theory ----
st.header("2. ทฤษฎีของอัลกอริทึม")

tabs = st.tabs(["Random Forest", "XGBoost", "SVM", "KNN", "Voting Ensemble"])

with tabs[0]:
    st.subheader("Random Forest")
    st.markdown("""
Random Forest คือ ensemble ของ Decision Trees จำนวนมาก โดย:
- แต่ละต้นใช้ข้อมูล **bootstrap sample** (สุ่มพร้อม replacement)
- แต่ละ node เลือก feature จาก **random subset** เท่านั้น (reduces overfitting)
- ผลลัพธ์สุดท้ายใช้ **majority vote** (classification)

**จุดเด่น:** ทนต่อ overfitting, ไม่ต้องปรับ scale, ดู feature importance ได้  
**Hyperparameters:** `n_estimators=200`, `max_depth=8`
    """)

with tabs[1]:
    st.subheader("XGBoost")
    st.markdown("""
XGBoost ใช้ **Gradient Boosting** — สร้างต้นไม้ทีละต้น โดยแต่ละต้นแก้ไขข้อผิดพลาดของต้นก่อน:

$$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$

โดย $h_m$ คือ weak learner ที่ fit กับ **residuals** ของรอบก่อน  
XGBoost เพิ่ม **L1/L2 regularization** ใน objective เพื่อป้องกัน overfitting

**Hyperparameters:** `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`
    """)

with tabs[2]:
    st.subheader("Support Vector Machine (SVM)")
    st.markdown("""
SVM หา **hyperplane** ที่แบ่งคลาสด้วย margin ที่ใหญ่ที่สุด:

$$\\min_{w,b} \\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i$$

ใช้ **RBF Kernel** เพื่อ map ข้อมูลไปยัง high-dimensional space:

$$K(x_i, x_j) = \\exp\\left(-\\gamma \\|x_i - x_j\\|^2\\right)$$

**จุดเด่น:** ทำงานได้ดีเมื่อ features น้อย, margin ชัดเจน  
**Hyperparameters:** `C=10`, `kernel='rbf'`, `gamma='scale'`
    """)

with tabs[3]:
    st.subheader("K-Nearest Neighbors (KNN)")
    st.markdown("""
KNN จำแนกโดยดูจาก K ตัวอย่างที่ใกล้ที่สุดใน feature space:

$$d(x_i, x_j) = \\sqrt{\\sum_k (x_{ik} - x_{jk})^2}$$

Vote จาก K เพื่อนบ้าน โดยใช้ **distance weighting** (เพื่อนที่ใกล้กว่า vote หนักกว่า)

**จุดเด่น:** ง่าย ไม่มี training, ยืดหยุ่น  
**Hyperparameters:** `k=7`, `weights='distance'`
    """)

with tabs[4]:
    st.subheader("Soft Voting Ensemble")
    st.markdown("""
รวม 4 โมเดลด้วย **Soft Voting** — เฉลี่ย predicted probability จากทุกโมเดล:

$$P(class=c) = \\frac{1}{4}\\sum_{m=1}^{4} P_m(class=c)$$

คลาสที่มี probability เฉลี่ยสูงที่สุดคือ prediction สุดท้าย

**ข้อดีเหนือ Hard Voting:** ใช้ความมั่นใจ (confidence) ของแต่ละโมเดลในการตัดสิน  
ทำให้ robust กว่าและ accuracy สูงกว่า
    """)

st.divider()

# ---- Development Steps ----
st.header("3. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
```
1. Load CSV → inspect shape, dtypes, missing values
2. Clean data → drop duplicates, impute, normalize text
3. Feature Engineering → weight_height_ratio, intelligence_tier
4. Encode → LabelEncoder for categoricals
5. Scale → StandardScaler (Z-score)
6. Split → train 80% / test 20% (stratified)
7. Build Ensemble → VotingClassifier(RF, XGB, SVM, KNN)
8. Train → ensemble.fit(X_train, y_train)
9. Evaluate → accuracy, classification_report, confusion_matrix
10. Save → joblib.dump(model, 'ensemble_model.pkl')
```
""")

st.divider()

# ---- References ----
st.header("4. แหล่งอ้างอิง")
st.markdown("""
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.
- Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Trans. IT*, 13(1).
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Dataset: https://www.kaggle.com/datasets/marshuu/dog-breeds
""")
