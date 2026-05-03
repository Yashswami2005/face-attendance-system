import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = "static/faces"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

print(f"🔍 Scanning folder: {BASE_DIR}")

for root, dirs, files in os.walk(BASE_DIR):
    label = os.path.basename(root)
    if label == "faces":
        continue

    print(f"➡️ Processing: {label}")

    for file in files:
        if file.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(root, file)
            print("   ✔", file)

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")

            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(label_ids[label])

if len(x_train) == 0:
    print("❌ NO FACES FOUND! Check your images.")
    exit()

print("\n📌 TRAINING STARTED...")
recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_trainer.yml")
print("✅ Saved face_trainer.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
print("✅ Saved labels.pickle")

print("\n🎉 TRAINING COMPLETE!")
