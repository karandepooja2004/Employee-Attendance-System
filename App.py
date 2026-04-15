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
from streamlit_autorefresh import st_autorefresh

# ------------------------------
# DATABASE CONNECTION
# ------------------------------

conn = mysql.connector.connect(
    host="82.180.143.66",
    user="u263681140_ADCET",
    password="Attendance@2026",
    database="u263681140_ADCET"
)

cursor = conn.cursor()
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


# ------------------------------
# Load CSV Files
# ------------------------------

if os.path.exists(person_csv):
    employees = pd.read_csv(person_csv)
else:
    employees = pd.DataFrame(columns=["Employee_ID","Name","Image_path"])

if os.path.exists(mapping_csv):
    employee_map = pd.read_csv(mapping_csv)
else:
    employee_map = pd.DataFrame(columns=["Employee_ID","Name"])

# ------------------------------
# Tabs
# ------------------------------

# tab1,tab2,tab3 = st.tabs([
#     "📷 Mark Attendance",
#     "🧑 Register Employee",
#     "📊 Attendance Report"
# ])
tab_choice = st.radio(
    "Navigation",
    ["📷 Mark Attendance", "🧑 Register Employee", "📊 Attendance Report"],
    horizontal=True
)

# =====================================================
# TAB 1 : ATTENDANCE (IN / OUT)
# =====================================================

if tab_choice == "📷 Mark Attendance":

    st.subheader("Auto Attendance System")

    st_autorefresh(interval=4000, key="refresh")

    FRAME_WINDOW = st.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Camera not working")

    else:
        ret, frame = cap.read()
        cap.release()

        if ret:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                faces = detector.detect_faces(image)
            except:
                faces = []

            detected_ids = []

            for face in faces:

                x,y,w,h = face['box']
                x,y = max(0,x), max(0,y)

                face_img = image[y:y+h, x:x+w]

                if face_img.size == 0:
                    continue

                face_img = cv2.resize(face_img,(160,160))

                embedding = embedder.embeddings([face_img])
                preds = svm_model.predict(embedding)
                prob = svm_model.predict_proba(embedding)

                confidence = np.max(prob)
                name = encoder.inverse_transform(preds)[0]

                if confidence >= threshold:

                    row = employee_map[employee_map["Name"]==name]

                    if not row.empty:
                        emp_id = row.iloc[0]["Employee_ID"]
                        detected_ids.append(emp_id)

                        label = f"{name} ({confidence*100:.2f}%)"
                        color = (0,255,0)
                    else:
                        label = "Match Not Found"
                        color = (0,0,255)
                else:
                    label = "Match Not Found"
                    color = (0,0,255)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

            FRAME_WINDOW.image(frame, channels="BGR")

            # =========================
            # ATTENDANCE DATABASE LOGIC
            # =========================

            detected_ids = list(set(detected_ids))

            if detected_ids:

                today = datetime.date.today().strftime("%Y-%m-%d")
                now = datetime.datetime.now()

                for emp_id in detected_ids:

                    row = employee_map[employee_map["Employee_ID"] == emp_id]
                    if row.empty:
                        continue

                    name = row.iloc[0]["Name"]

                    cursor.execute("""
                        SELECT IN_Time, OUT_Time FROM attendance
                        WHERE Employee_ID=%s AND Date=%s
                    """, (str(emp_id), today))

                    record = cursor.fetchone()

                    # -------- IN --------
                    if record is None:

                        cursor.execute("""
                            INSERT INTO attendance 
                            (Employee_ID, Name, Date, IN_Time, OUT_Time, Hours)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """, (
                            str(emp_id),
                            str(name),
                            today,
                            now.strftime("%H:%M:%S"),
                            None,
                            0
                        ))

                        st.success(f"{name} IN marked")

                    # -------- OUT --------
                    else:

                        in_time, out_time = record

                        if out_time is None:

                            in_dt = datetime.datetime.strptime(str(in_time), "%H:%M:%S")
                            out_dt = datetime.datetime.strptime(now.strftime("%H:%M:%S"), "%H:%M:%S")

                            worked_seconds = (out_dt - in_dt).total_seconds()

                            if worked_seconds < 0:
                                worked_seconds += 24*3600

                            worked_hours = worked_seconds / 3600

                            # 🔥 CONDITION
                            if worked_hours < 4:
                                st.warning(f"{name} ❗ Minimum 4 hours required")
                            else:
                                cursor.execute("""
                                    UPDATE attendance
                                    SET OUT_Time=%s, Hours=%s
                                    WHERE Employee_ID=%s AND Date=%s
                                """, (
                                    now.strftime("%H:%M:%S"),
                                    round(worked_hours,2),
                                    str(emp_id),
                                    today
                                ))

                                st.success(f"{name} OUT marked")

                conn.commit()

# =====================================================
# TAB 2 : REGISTER EMPLOYEE (UNCHANGED)
# =====================================================

elif tab_choice == "🧑 Register Employee":

    st.subheader("Register New Student")

    student_id = st.text_input("Employee ID")
    name = st.text_input("Employee Name")

    if st.button("Start Image Capture"):

        if student_id and name:

            dataset_path = "dataset/" + name
            os.makedirs(dataset_path,exist_ok=True)

            cap = cv2.VideoCapture(0)
            # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                st.error("Camera could not be opened")
            else:

                count = 0
                max_images = 100

                st.info("Press 'c' to capture image. Capture at least 70 images.")

                while True:

                    ret, frame = cap.read()

                    if not ret or frame is None:
                        break

                    faces = []

                    try:
                        faces = detector.detect_faces(frame)
                    except:
                        faces = []

                    face_img = None

                    for face in faces:

                        x,y,w,h = face['box']

                        x = max(0,x)
                        y = max(0,y)

                        face_img = frame[y:y+h,x:x+w]

                        if face_img.size == 0:
                            continue

                        face_img = cv2.resize(face_img,(160,160))

                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                        cv2.imshow("Face Preview",face_img)

                    cv2.putText(frame,"Look Straight / Left / Right / Up / Down",
                                (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,255),
                                2)

                    cv2.putText(frame,f"Images Captured: {count}/{max_images}",
                                (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255,0,0),
                                2)

                    cv2.imshow("Dataset Creator",frame)

                    key = cv2.waitKey(1)

                    if key == ord('c') and len(faces) > 0 and face_img is not None:

                        img_path = dataset_path + "/" + str(count) + ".jpg"

                        cv2.imwrite(img_path,face_img)

                        count += 1

                    if count >= max_images:
                        break

                    if key == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

                st.success("Image Capture Completed")

    # ------------------------------
    # SUBMIT REGISTRATION
    # ------------------------------

    if st.button("Submit Registration"):

        if student_id and name:

            dataset_path = "dataset/" + name

            if not os.path.exists(dataset_path):
                st.error("Capture images first")
            else:

                images = os.listdir(dataset_path)

                new_rows = []

                for img in images:

                    path = dataset_path + "/" + img

                    new_rows.append({
                        "Employee_ID":student_id,
                        "Name":name,
                        "Image_path":path
                    })

                new_df = pd.DataFrame(new_rows)

                employees_updated = pd.concat([employees,new_df],ignore_index=True)

                employees_updated.to_csv(person_csv,index=False)

                new_map = pd.DataFrame([{
                    "Employee_ID":student_id,
                    "Name":name
                }])

                employee_map_updated = pd.concat([employee_map,new_map],ignore_index=True)

                employee_map_updated.to_csv(mapping_csv,index=False)

                st.success("Student Registered Successfully")

# =====================================================
# TAB 3 : ATTENDANCE REPORT
# =====================================================

elif tab_choice == "📊 Attendance Report":

    st.subheader("Employee Attendance Report")

    cursor.execute("""
        SELECT Employee_ID, Name, Date, IN_Time, OUT_Time, Hours 
        FROM attendance 
        ORDER BY Date DESC
    """)

    rows = cursor.fetchall()

    columns = ["Employee_ID","Name","Date","IN Time","OUT Time","Hours"]

    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:

        # ✅ FIX IN TIME (timedelta → HH:MM)
        df["IN Time"] = df["IN Time"].apply(
            lambda x: str(x).split(" ")[-1][:5] if x else "-"
        )

        # ✅ FIX OUT TIME
        df["OUT Time"] = df["OUT Time"].apply(
            lambda x: str(x).split(" ")[-1][:5] if x else "-"
        )

        # ✅ FIX HOURS (CRITICAL)
        def fix_hours(x):
            try:
                x = float(x)
                if x > 100:  # microseconds/seconds issue
                    return round(x / 3600, 2)
                return round(x, 2)
            except:
                return 0

        df["Hours"] = df["Hours"].apply(fix_hours)

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("No attendance yet")