from flask import Flask, render_template, Response, request, redirect, url_for, session, send_file 
import cv2
import pickle
import sqlite3
import datetime
import os
import io
import csv
from functools import wraps
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_key_please"

# ---------- Admin credentials ----------
ADMIN_USERNAME = "Yashswami"
ADMIN_PASSWORD = "Yash@2005"
# --------------------------------------

# ✅ Load recognizer safely (global variable)
recognizer = None
try:
    if os.path.exists("face_trainer.yml"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_trainer.yml")
    else:
        recognizer = None
        print("⚠️ face_trainer.yml NOT found")
except Exception as e:
    recognizer = None
    print("⚠️ cv2.face recognizer not available:", e)

# ✅ Load label mappings (id -> label)
labels = {}
if os.path.exists("labels.pickle"):
    with open("labels.pickle", "rb") as f:
        raw = pickle.load(f)            # raw: {label: id}
        labels = {v: k for k, v in raw.items()}  # convert to {id: label}
else:
    print("⚠️ labels.pickle NOT found")

# ✅ Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Initialize Database
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    # Attendance table
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT,
            name TEXT,
            branch TEXT,
            division TEXT,
            date TEXT,
            hour TEXT,
            time TEXT
        )
    """)

    # Students table (new)
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE,
            name TEXT,
            branch TEXT,
            division TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------- Helper: Login required ----------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrap

# ---------- Helper: Mark attendance ----------
def mark_attendance(roll_no=None, name=None, branch="N/A", division="N/A"):
    if not name:
        return

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    # 🔹 Try to fetch actual Roll No, Branch & Division from students table
    try:
        if roll_no and roll_no != "N/A":
            c.execute("SELECT roll_no, branch, division FROM students WHERE roll_no = ?", (roll_no,))
        else:
            c.execute("SELECT roll_no, branch, division FROM students WHERE name = ?", (name,))
        student = c.fetchone()
        if student:
            roll_no, branch, division = student
    except sqlite3.OperationalError:
        pass  # if table not found, skip safely

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    time = now.strftime("%H:%M:%S")

    # 🔹 Avoid duplicate attendance per hour
    c.execute("SELECT 1 FROM attendance WHERE roll_no=? AND date=? AND hour=?", (roll_no, date, hour))
    if c.fetchone() is None:
        c.execute("""
            INSERT INTO attendance (roll_no, name, branch, division, date, hour, time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (roll_no, name, branch, division, date, hour, time))
        conn.commit()
        print(f"✅ Attendance marked for {name} ({roll_no}) - {branch}-{division}")
    else:
        print(f"ℹ️ {name} ({roll_no}) already marked this hour")

    conn.close()


# ---------- Helper: Auto face recognition ----------
def generate_frames():
    """
    Yields multipart JPEG frames for Flask streaming.
    Uses a stricter confidence threshold and resizes ROI to match training size.
    """
    global recognizer, labels
    seen = set()
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        hour = now.strftime("%H")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # resize to training size (200x200) — LBPH expects consistent sizes
            try:
                roi_gray_resized = cv2.resize(roi_gray, (200, 200))
            except Exception:
                roi_gray_resized = roi_gray

            name = "Unknown"
            roll_no = "Unknown"
            branch, division = "N/A", "N/A"
            color = (0, 0, 255)

            if recognizer is not None:
                try:
                    id_, conf = recognizer.predict(roi_gray_resized)
                except Exception:
                    id_, conf = None, 1000

                # STRONGER threshold: 45 (tune slightly if needed)
                if id_ is not None and conf < 45:
                    label = labels.get(id_, "Unknown")  # label is folder like "84_Yash Swami"
                    if "_" in label:
                        roll_no, name = label.split("_", 1)
                    else:
                        name = label
                        roll_no = "N/A"

                    color = (0, 255, 0)
                    key = (roll_no, date, hour)
                    if roll_no not in (None, "Unknown", "N/A") and key not in seen:
                        mark_attendance(roll_no, name, branch, division)
                        seen.add(key)
                else:
                    # keep Unknown / no marking
                    pass

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------- Helper: Train Model ----------
def train_model():
    """
    Re-trains the LBPH model using images in static/faces/<roll_name>/img_*.jpg
    - folder name is used as the label (e.g. "84_Yash Swami")
    - images are assumed to be face crops; we resize them to 200x200
    - saves face_trainer.yml and labels.pickle
    - updates the global recognizer and labels dict so the running app uses new model
    """
    global recognizer, labels
    base_dir = "static/faces"
    new_recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = {}
    y_labels = []
    x_train = []
    current_id = 0

    if not os.path.exists(base_dir):
        print("⚠️ faces directory not found:", base_dir)
        return

    # iterate over each student folder
    for dirname in sorted(os.listdir(base_dir)):
        student_path = os.path.join(base_dir, dirname)
        if not os.path.isdir(student_path):
            continue

        label = dirname  # folder name like "84_Yash Swami"
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        # read images in that folder
        for file in os.listdir(student_path):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                path = os.path.join(student_path, file)
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # image is already a face crop from capture; resize to uniform size
                    roi = cv2.resize(gray, (200, 200))
                    x_train.append(roi)
                    y_labels.append(id_)
                except Exception as e:
                    print("⚠️ Error reading", path, e)
                    continue

    if len(x_train) > 0:
        new_recognizer.train(x_train, np.array(y_labels))
        new_recognizer.save("face_trainer.yml")
        with open("labels.pickle", "wb") as f:
            pickle.dump(label_ids, f)

        # update globals so running app uses new model immediately
        recognizer = new_recognizer
        labels = {v: k for k, v in label_ids.items()}
        print("✅ Model retrained successfully")
    else:
        print("⚠️ No faces found for training")

# ---------- Route: Capture Faces ----------
@app.route("/capture/<roll_no>/<name>/<branch>/<division>")
@login_required
def capture(roll_no, name, branch, division):
    folder_name = f"{roll_no}_{name}"
    folder = os.path.join("static/faces", folder_name)
    os.makedirs(folder, exist_ok=True)

    # 🔹 Save student details into students table
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO students (roll_no, name, branch, division) VALUES (?, ?, ?, ?)",
              (roll_no, name, branch, division))
    conn.commit()
    conn.close()

    camera = cv2.VideoCapture(0)
    count = 0
    target = 60  # capture 60 images

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            # SAVE GRAYSCALE CROP (uniform size)
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (200, 200))
            except Exception:
                # fallback: skip this frame if resize fails
                count -= 1
                continue

            cv2.imwrite(f"{folder}/img_{count:03}.jpg", face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/{target}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Auto Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif count >= target:
            break

    camera.release()
    cv2.destroyAllWindows()

    # retrain model after capture
    train_model()

    return f"<h3>✅ Captured {target} images for {name} ({roll_no}) - {branch}-{division} and retrained!</h3><a href='/admin'>Back to Dashboard</a>"

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USERNAME and request.form["password"] == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid credentials")
    return render_template("admin_login.html")

@app.route("/admin/logout")
@login_required
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.route("/admin")
@login_required
def admin_dashboard():
    return render_template("admin_dashboard.html")

@app.route("/add_student", methods=["GET", "POST"])
@login_required
def add_student():
    if request.method == "POST":
        name = request.form["name"]
        roll_no = request.form["roll_no"]
        branch = request.form["branch"]
        division = request.form["division"]
        return redirect(url_for("capture", roll_no=roll_no, name=name, branch=branch, division=division))
    return render_template("add_student.html")

@app.route("/attendance")
@login_required
def attendance():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT roll_no, name, branch, division, date, hour, time FROM attendance ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return render_template("attendance.html", records=records)

@app.route("/export_csv")
@login_required
def export_csv():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT roll_no, name, branch, division, date, hour, time FROM attendance ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Roll No", "Name", "Branch", "Division", "Date", "Hour", "Time"])
    cw.writerows(rows)
    mem = io.BytesIO(si.getvalue().encode("utf-8"))
    mem.seek(0)
    filename = f"attendance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, as_attachment=True, download_name=filename, mimetype="text/csv")

@app.route("/shutdown")
def shutdown():
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return "Camera released."

if __name__ == "__main__":
    app.run(debug=True)
