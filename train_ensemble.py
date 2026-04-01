"""
Ensemble ML Model — Dog Breed Group Classification
โมเดลที่ 1: Voting Classifier ประกอบจาก 4 โมเดล
- RandomForest
- XGBoost
- SVM
- KNN
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Import preprocessing pipeline
from preprocess_csv import (
    load_and_inspect, clean_data, engineer_features,
    encode_and_scale, save_artifacts
)


def build_ensemble():
    """
    สร้าง Voting Classifier จาก 4 โมเดล
    ใช้ soft voting (เฉลี่ย probability)
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    svm = SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        probability=True,   # จำเป็นสำหรับ soft voting
        random_state=42
    )

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[
            ("rf",  rf),
            ("xgb", xgb),
            ("svm", svm),
            ("knn", knn),
        ],
        voting="soft",       # เฉลี่ย probability จากทุกโมเดล
        n_jobs=-1
    )

    return ensemble


def evaluate(model, X_test, y_test, class_names):
    """แสดงผลการประเมินโมเดล"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Accuracy: {acc:.4f} ===\n")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return acc, y_pred


def save_model(model, path="models/ensemble_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Save] Ensemble model saved → {path}")


# ---- Run ----
if __name__ == "__main__":
    # 1. โหลดและเตรียมข้อมูล
    df = load_and_inspect("datasets/dog_breeds_stats.csv")
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, encoders = encode_and_scale(df, target_col="Country of Origin")
    save_artifacts(scaler, encoders)

    class_names = encoders["target"].classes_

    # 2. สร้างและเทรนโมเดล
    print("\n[Train] Building ensemble model...")
    ensemble = build_ensemble()
    ensemble.fit(X_train, y_train)
    print("[Train] Done!")

    # 3. ประเมินผล
    acc, _ = evaluate(ensemble, X_test, y_test, None)

    # 4. บันทึกโมเดล
    save_model(ensemble)
