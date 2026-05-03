import cv2
import os
import numpy as np
import pickle

faces_dir = "static/faces"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

label_ids = {}
x_train = []
y_labels = []
current_id = 0

# FIXED: folder name = student name
for root, dirs, files in os.walk(faces_dir):
    for dirname in dirs:
        student_path = os.path.join(faces_dir, dirname)
        label = dirname  # folder name = student name

        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1
        
        id_ = label_ids[label]

        # read images from folder
        for file in os.listdir(student_path):
            if file.lower().endswith(("jpg", "png", "jpeg")):
                path = os.path.join(student_path, file)

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # detect face
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]

                    # FIXED: resize face for uniform training
                    roi = cv2.resize(roi, (200, 200))

                    x_train.append(roi)
                    y_labels.append(id_)

# save model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_trainer.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("🎉 Training DONE!")
print("✔ Model saved as face_trainer.yml")
print("✔ Labels saved as labels.pickle")
