import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
from PIL import Image
import pandas as pd
import datetime
import pytz
import mysql.connector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ------------------------------
# DATABASE CONNECTION
# ------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="82.180.143.66",
        user="u263681140_ADCET",
        password="Attendance@2026",
        database="u263681140_ADCET"
    )

# ------------------------------
# Page Config & Models
# ------------------------------
st.set_page_config(page_title="WorkTrack AI", page_icon="🏢", layout="wide")

@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    svm_model = joblib.load("models/svm_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return detector, embedder, svm_model, encoder

detector, embedder, svm_model, encoder = load_models()
threshold = 0.60
INDIAN_TZ = pytz.timezone('Asia/Kolkata')

# ------------------------------
# CSV Paths & Loading
# ------------------------------
person_csv = "Person_Info_New.csv"
mapping_csv = "models/employee_mapping.csv"

def load_csv_data():
    if os.path.exists(person_csv):
        emps = pd.read_csv(person_csv)
    else:
        emps = pd.DataFrame(columns=["Employee_ID","Name","Image_path"])
    
    if os.path.exists(mapping_csv):
        m_map = pd.read_csv(mapping_csv)
    else:
        m_map = pd.DataFrame(columns=["Employee_ID","Name"])
    return emps, m_map

employees, employee_map = load_csv_data()

# ------------------------------
# Navigation
# ------------------------------
tab_choice = st.radio("Navigation", ["📷 Mark Attendance", "🧑 Register Employee", "📊 Attendance Report"], horizontal=True)

# =====================================================
# TAB 1 : MARK ATTENDANCE (Live Camera Recognition)
# =====================================================
if tab_choice == "📷 Mark Attendance":
    st.subheader("Auto Attendance System - Live Feed")

    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_processed = {}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Recognition needs RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                faces = detector.detect_faces(rgb_img)
            except:
                faces = []

            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                face_img = rgb_img[y:y+h, x:x+w]

                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (160, 160))
                    embedding = embedder.embeddings([face_img])
                    preds = svm_model.predict(embedding)
                    prob = svm_model.predict_proba(embedding)
                    
                    confidence = np.max(prob)
                    name = encoder.inverse_transform(preds)[0]

                    if confidence >= threshold:
                        color = (0, 255, 0)
                        label = f"{name} ({confidence*100:.2f}%)"
                        # MARK ATTENDANCE LOGIC
                        self.process_attendance_logic(name)
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"

                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def process_attendance_logic(self, name):
            # 30-second throttle per person to prevent DB spam
            now_utc = datetime.datetime.now()
            if name in self.last_processed and (now_utc - self.last_processed[name]).seconds < 30:
                return

            try:
                db = get_db_connection()
                cursor = db.cursor()
                
                now_local = datetime.datetime.now(INDIAN_TZ)
                today = now_local.strftime("%Y-%m-%d")
                current_time_str = now_local.strftime("%H:%M:%S")

                row = employee_map[employee_map["Name"] == name]
                if not row.empty:
                    emp_id = str(row.iloc[0]["Employee_ID"])
                    
                    cursor.execute("SELECT IN_Time, OUT_Time FROM attendance WHERE Employee_ID=%s AND Date=%s", (emp_id, today))
                    record = cursor.fetchone()

                    # -------- LOGIC: IN --------
                    if record is None:
                        cursor.execute("""
                            INSERT INTO attendance (Employee_ID, Name, Date, IN_Time, OUT_Time, Hours)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """, (emp_id, name, today, current_time_str, None, 0))
                        db.commit()
                        self.last_processed[name] = now_utc
                    
                    # -------- LOGIC: OUT --------
                    else:
                        in_time, out_time = record
                        if out_time is None:
                            in_dt = datetime.datetime.strptime(str(in_time), "%H:%M:%S")
                            out_dt = datetime.datetime.strptime(current_time_str, "%H:%M:%S")
                            
                            worked_seconds = (out_dt - in_dt).total_seconds()
                            if worked_seconds < 0: worked_seconds += 24*3600
                            worked_hours = worked_seconds / 3600

                            # Your 4-hour minimum check
                            if worked_hours >= 4:
                                cursor.execute("""
                                    UPDATE attendance SET OUT_Time=%s, Hours=%s
                                    WHERE Employee_ID=%s AND Date=%s
                                """, (current_time_str, round(worked_hours, 2), emp_id, today))
                                db.commit()
                                self.last_processed[name] = now_utc
                db.close()
            except Exception as e:
                pass # Silently fail in thread to prevent video freeze

    webrtc_streamer(
        key="attendance-live",
        video_processor_factory=AttendanceProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# =====================================================
# TAB 2 : REGISTER EMPLOYEE (UNCHANGED)
# =====================================================
elif tab_choice == "🧑 Register Employee":
    st.subheader("Register New Employee")
    
    student_id = st.text_input("Employee ID")
    name = st.text_input("Employee Name")
    img_file = st.camera_input("Take reference photo")

    if st.button("Submit Registration"):
        if student_id and name and img_file:
            dataset_path = f"dataset/{name}"
            os.makedirs(dataset_path, exist_ok=True)
            
            img = Image.open(img_file)
            frame = np.array(img)
            path = f"{dataset_path}/0.jpg"
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            new_rows = [{"Employee_ID": student_id, "Name": name, "Image_path": path}]
            updated_employees = pd.concat([employees, pd.DataFrame(new_rows)], ignore_index=True)
            updated_employees.to_csv(person_csv, index=False)

            new_map = [{"Employee_ID": student_id, "Name": name}]
            updated_map = pd.concat([employee_map, pd.DataFrame(new_map)], ignore_index=True)
            updated_map.to_csv(mapping_csv, index=False)
            
            st.success("Registration Successful!")

# =====================================================
# TAB 3 : ATTENDANCE REPORT (UNCHANGED)
# =====================================================
elif tab_choice == "📊 Attendance Report":
    st.subheader("Employee Attendance Report")

    try:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT Employee_ID, Name, Date, IN_Time, OUT_Time, Hours FROM attendance ORDER BY Date DESC")
        rows = cursor.fetchall()
        db.close()

        columns = ["Employee_ID","Name","Date","IN Time","OUT Time","Hours"]
        df = pd.DataFrame(rows, columns=columns)

        if not df.empty:
            df["IN Time"] = df["IN Time"].apply(lambda x: str(x).split(" ")[-1][:5] if x else "-")
            df["OUT Time"] = df["OUT Time"].apply(lambda x: str(x).split(" ")[-1][:5] if x else "-")

            def fix_hours(x):
                try:
                    x = float(x)
                    if x > 100: return round(x / 3600, 2)
                    return round(x, 2)
                except: return 0

            df["Hours"] = df["Hours"].apply(fix_hours)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No attendance yet")
    except Exception as e:
        st.error(f"Error: {e}")
