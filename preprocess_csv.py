"""
Data Preprocessing — Dataset 1: Dog Breeds Stats (Structured CSV)
ที่มา: Kaggle — Dog Breeds Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_inspect(path="datasets/dog_breeds_stats.csv"):
    """โหลดและตรวจสอบข้อมูลเบื้องต้น"""
    df = pd.read_csv(path)
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== dtypes ===")
    print(df.dtypes)
    print("\n=== Sample ===")
    print(df.head())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ขั้นตอนการทำความสะอาดข้อมูล
    - จัดการ missing values
    - แก้ไข data types
    - ลบ duplicates
    """
    df = df.copy()

    # 1. ลบ duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Clean] Removed {before - len(df)} duplicate rows")

    # 2. จัดการ missing values (numeric) — เติมด้วย median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"[Clean] Filled {n_missing} missing in '{col}' with median={df[col].median():.2f}")

    # 3. จัดการ missing values (categorical) — เติมด้วย mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[Clean] Filled {n_missing} missing in '{col}' with mode='{mode_val}'")

    # 4. แก้ไข text columns (strip whitespace, lowercase)
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering
    - สร้าง features ใหม่จากข้อมูลที่มี
    """
    df = df.copy()

    # BMI-like ratio: weight / height
    if "weight_kg" in df.columns and "height_cm" in df.columns:
        df["weight_height_ratio"] = df["weight_kg"] / (df["height_cm"] + 1e-5)

    # Intelligence tier (แบ่ง rank เป็น 3 ระดับ)
    if "intelligence_rank" in df.columns:
        df["intelligence_tier"] = pd.cut(
            df["intelligence_rank"],
            bins=[0, 30, 60, 150],
            labels=["high", "medium", "low"]
        )

    return df


def encode_and_scale(df: pd.DataFrame, target_col="Country of Origin"):
    """
    Encode categorical → numeric
    Scale numeric features
    Return: X_train, X_test, y_train, y_test, scaler, encoders
    """
    df = df.copy()

    encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # เอา target ออกจาก cat_cols
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # Label encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))
    encoders["target"] = le_target
    df = df.drop(columns=[target_col])

    # Drop breed name column ถ้ามี (ใช้เป็น identifier ไม่ใช่ feature)
    if "breed" in df.columns:
        df = df.drop(columns=["breed"])

    X = df.values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
    	X, y, test_size=0.2, random_state=42
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"[Encode] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[Encode] Classes: {le_target.classes_}")

    return X_train, X_test, y_train, y_test, scaler, encoders


def save_artifacts(scaler, encoders, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    joblib.dump(encoders, f"{out_dir}/encoders.pkl")
    print(f"[Save] Artifacts saved to {out_dir}/")


# ---- Run ----
if __name__ == "__main__":
    df = load_and_inspect()
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, encoders = encode_and_scale(df)
    save_artifacts(scaler, encoders)
