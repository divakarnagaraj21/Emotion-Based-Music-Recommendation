# ============================================================
# 🤖 EMOTION-BASED MUSIC RECOMMENDATION SYSTEM
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np, os, cv2, random, json, matplotlib.pyplot as plt

# -------------------------------
# 1. PATHS & CONFIG
# -------------------------------
TRAIN_DIR = "emotion_dataset/train"   # e.g. train/happy/, train/sad/, train/angry/, etc.
VAL_DIR = "emotion_dataset/val"
MODEL_PATH = "emotion_model.keras"
LABEL_PATH = "emotion_labels.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

os.makedirs("models", exist_ok=True)

print("✅ TensorFlow version:", tf.__version__)

# -------------------------------
# 2. DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save labels
labels = {v: k for k, v in train_gen.class_indices.items()}
with open(LABEL_PATH, "w") as f:
    json.dump(labels, f)
print(f"✅ Saved emotion labels to {LABEL_PATH}")

# -------------------------------
# 3. MODEL BUILDING (Transfer Learning)
# -------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 4. TRAINING
# -------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
]

print("\n🚀 Training emotion recognition model...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# -------------------------------
# 5. REAL-TIME EMOTION DETECTION + MUSIC RECOMMENDATION
# -------------------------------
# A simple emotion-to-music mapping
music_dict = {
    "happy": ["Happy.mp3", "GoodVibes.mp3"],
    "sad": ["SomeoneLikeYou.mp3", "FixYou.mp3"],
    "angry": ["Believer.mp3", "Thunderstruck.mp3"],
    "neutral": ["LetItBe.mp3", "Perfect.mp3"]
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model(MODEL_PATH)
labels = json.load(open(LABEL_PATH))

print("\n🎵 Starting real-time emotion-based music suggestion...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, IMG_SIZE)
        face_norm = np.expand_dims(face_resized / 255.0, axis=0)

        pred = model.predict(face_norm)
        emotion = labels[str(np.argmax(pred))]
        
        # Display emotion
        cv2.putText(frame, f"Emotion: {emotion.upper()}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Recommend a song (random)
        suggested_song = random.choice(music_dict.get(emotion, ["No suggestion"]))
        cv2.putText(frame, f"🎵 {suggested_song}", (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Emotion-Based Music Recommender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
