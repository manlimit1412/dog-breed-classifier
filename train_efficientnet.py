"""
Neural Network Model — Dog Breed Image Classification
โมเดลที่ 2: EfficientNetB0 (Pretrained) + Fine-tuning
Dataset: Dog Breed Image Dataset (70 breeds) จาก Kaggle
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- Config ----
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 70
EPOCHS_FROZEN = 10     # Phase 1: เทรนเฉพาะ head
EPOCHS_FINETUNE = 10   # Phase 2: fine-tune ชั้นบน
TRAIN_DIR = "datasets/images/train"
TEST_DIR  = "datasets/images/test"
MODEL_SAVE_PATH = "models/efficientnet_model.h5"


# ---- Data Generators ----
def make_generators():
    """
    สร้าง ImageDataGenerator
    - Train: augmentation (flip, rotate, zoom, brightness)
    - Val/Test: rescale เท่านั้น
    """
    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=0.15       # แบ่ง 15% จาก train เป็น val
    )

    test_gen = ImageDataGenerator(rescale=1.0/255)

    train_ds = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_ds = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    test_ds = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_ds, val_ds, test_ds


# ---- Model Architecture ----
def build_model(num_classes: int, freeze_base=True):
    """
    EfficientNetB0 + Custom Classification Head
    
    Architecture:
    Input (224x224x3)
      → EfficientNetB0 (pretrained ImageNet, frozen)
      → GlobalAveragePooling2D
      → BatchNormalization
      → Dense(256, relu)
      → Dropout(0.4)
      → Dense(128, relu)
      → Dropout(0.3)
      → Dense(num_classes, softmax)
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,          # ไม่เอา classification head เดิม
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = not freeze_base  # Phase 1: freeze

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def get_callbacks(phase="frozen"):
    """Callbacks สำหรับการเทรน"""
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f"models/efficientnet_{phase}_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]


def plot_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"models/{title.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"[Plot] Saved training history.")


# ---- Main Training Pipeline ----
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # 1. สร้าง data generators
    print("[Data] Creating generators...")
    train_ds, val_ds, test_ds = make_generators()
    n_classes = len(train_ds.class_indices)
    print(f"[Data] {n_classes} classes detected")

    # บันทึก class mapping
    import json
    with open("models/class_indices.json", "w") as f:
        json.dump(train_ds.class_indices, f, indent=2)

    # 2. Phase 1: เทรนเฉพาะ head (base frozen)
    print("\n[Phase 1] Training classification head (base model frozen)...")
    model, base_model = build_model(n_classes, freeze_base=True)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history1 = model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=get_callbacks("frozen")
    )
    plot_history(history1, "Phase 1 Frozen")

    # 3. Phase 2: Fine-tune ชั้น top ของ base model
    print("\n[Phase 2] Fine-tuning top layers of EfficientNetB0...")
    base_model.trainable = True

    # Freeze ชั้นล่าง, เปิดเฉพาะ 30 ชั้นบนสุด
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),   # LR ต่ำมากสำหรับ fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=get_callbacks("finetune")
    )
    plot_history(history2, "Phase 2 Fine-tune")

    # 4. ประเมินบน test set
    print("\n[Eval] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[Eval] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # 5. บันทึกโมเดล
    model.save(MODEL_SAVE_PATH)
    print(f"[Save] Model saved → {MODEL_SAVE_PATH}")
