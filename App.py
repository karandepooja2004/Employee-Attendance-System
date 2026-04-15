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
import mysql.connector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ------------------------------
# DATABASE CONNECTION
# ------------------------------
# Use a function to handle connection to avoid "Lost connection" errors
def get_db_connection():
    return mysql.connector.connect(
        host="82.180.143.66",
        user="u263681140_ADCET",
        password="Attendance@2026",
        database="u263681140_ADCET"
    )

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="WorkTrack AI - Employee Attendance",
    page_icon="🏢",
    layout="wide"
)

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    svm_model = joblib.load("models/svm_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return detector, embedder, svm_model, encoder

detector, embedder, svm_model, encoder = load_models()
threshold = 0.60

# ------------------------------
# CSV Paths
# ------------------------------
person_csv = "Person_Info_New.csv"
mapping_csv = "models/employee_mapping.csv"

# Load CSV Files
if os.path.exists(person_csv):
    employees = pd.read_csv(person_csv)
else:
    employees = pd.DataFrame(columns=["Employee_ID","Name","Image_path"])

if os.path.exists(mapping_csv):
    employee_map = pd.read_csv(mapping_csv)
else:
    employee_map = pd.DataFrame(columns=["Employee_ID","Name"])

# ------------------------------
# Navigation
# ------------------------------
tab_choice = st.sidebar.radio(
    "Navigation",
    ["📷 Mark Attendance", "🧑 Register Employee", "📊 Attendance Report"]
)

# =====================================================
# TAB 1 : ATTENDANCE (WebRTC Version)
# =====================================================
if tab_choice == "📷 Mark Attendance":
    st.subheader("Auto Attendance System")
    st.info("The system will automatically detect your face and mark attendance in the database.")

    class AttendanceProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                faces = detector.detect_faces(rgb_img)
            except:
                faces = []

            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                face_crop = rgb_img[y:y+h, x:x+w]

                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (160, 160))
                    embedding = embedder.embeddings([face_crop])
                    
                    preds = svm_model.predict(embedding)
                    prob = svm_model.predict_proba(embedding)
                    confidence = np.max(prob)
                    name = encoder.inverse_transform(preds)[0]

                    if confidence >= threshold:
                        color = (0, 255, 0) # Green
                        label = f"{name} ({confidence*100:.1f}%)"
                        
                        # Database logic inside a helper to avoid threading crashes
                        self.log_attendance(name)
                    else:
                        color = (0, 0, 255) # Red
                        label = "Unknown"

                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def log_attendance(self, name):
            try:
                temp_conn = get_db_connection()
                temp_cursor = temp_conn.cursor()
                today = datetime.date.today().strftime("%Y-%m-%d")
                now = datetime.datetime.now()

                # Get Emp ID from map
                row = employee_map[employee_map["Name"] == name]
                if not row.empty:
                    emp_id = str(row.iloc[0]["Employee_ID"])
                    
                    temp_cursor.execute("SELECT IN_Time, OUT_Time FROM attendance WHERE Employee_ID=%s AND Date=%s", (emp_id, today))
                    record = temp_cursor.fetchone()

                    if record is None:
                        temp_cursor.execute("""
                            INSERT INTO attendance (Employee_ID, Name, Date, IN_Time, OUT_Time, Hours)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """, (emp_id, name, today, now.strftime("%H:%M:%S"), None, 0))
                    
                    temp_conn.commit()
                temp_conn.close()
            except:
                pass

    webrtc_streamer(key="attendance", video_processor_factory=AttendanceProcessor, 
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# =====================================================
# TAB 2 : REGISTER EMPLOYEE (Web-Friendly Version)
# =====================================================
elif tab_choice == "🧑 Register Employee":
    st.subheader("Register New Employee")
    
    col1, col2 = st.columns(2)
    with col1:
        reg_id = st.text_input("Employee ID")
    with col2:
        reg_name = st.text_input("Employee Name")

    st.write("### Capture Training Photos")
    img_file = st.camera_input("Take a clear photo of the employee")

    if img_file:
        if reg_id and reg_name:
            # Save logic
            dataset_path = f"dataset/{reg_name}"
            os.makedirs(dataset_path, exist_ok=True)
            
            # Convert to OpenCV format to detect face before saving
            img = Image.open(img_file)
            frame = np.array(img)
            
            faces = detector.detect_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                face_img = frame[max(0,y):y+h, max(0,x):x+w]
                face_img = cv2.resize(face_img, (160, 160))
                
                # We save one high-quality image. Usually, for SVM, 1-5 good images are enough.
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{dataset_path}/img_{timestamp}.jpg"
                cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                st.success(f"Image Captured and saved to {save_path}")
                
                if st.button("Complete Registration"):
                    # Update CSVs
                    new_entry = pd.DataFrame([{"Employee_ID": reg_id, "Name": reg_name, "Image_path": save_path}])
                    employees = pd.concat([employees, new_entry], ignore_index=True)
                    employees.to_csv(person_csv, index=False)
                    
                    new_map = pd.DataFrame([{"Employee_ID": reg_id, "Name": reg_name}])
                    employee_map = pd.concat([employee_map, new_map], ignore_index=True)
                    employee_map.to_csv(mapping_csv, index=False)
                    
                    st.balloons()
                    st.success("Registration Permanent!")
            else:
                st.error("No face detected! Please look directly at the camera.")
        else:
            st.warning("Please enter ID and Name first.")

# =====================================================
# TAB 3 : ATTENDANCE REPORT
# =====================================================
elif tab_choice == "📊 Attendance Report":
    st.subheader("Employee Attendance Report")
    
    try:
        db = get_db_connection()
        query = "SELECT Employee_ID, Name, Date, IN_Time, OUT_Time, Hours FROM attendance ORDER BY Date DESC"
        df_report = pd.read_sql(query, db)
        db.close()

        if not df_report.empty:
            st.dataframe(df_report, use_container_width=True)
        else:
            st.warning("No records found in database.")
    except Exception as e:
        st.error(f"Database Error: {e}")
