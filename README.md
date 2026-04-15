# 🎓 WorkTrack AI - Employee Attendance System

An AI-powered Employee Attendance System using **Face Recognition** that automatically marks **IN/OUT time** and calculates **working hours**.

---

## 🚀 Features

- Face Recognition based attendance
- Automatic IN / OUT time detection
- Working hours calculation
- Minimum 4-hour validation before OUT
- Real-time attendance using webcam
- MySQL database integration
- Employee registration with dataset creation
- Live attendance report dashboard 

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MTCNN (Face Detection)
- FaceNet (Face Embeddings)
- SVM (Face Recognition)
- Streamlit (Frontend UI)
- MySQL (Database)

---

## ▶️ Run Project

    streamlit run app.py

---

## 🧑‍💼 How It Works

- Register employee → Capture face images
- Train model → Generate embeddings
- System detects face via webcam
- Marks:
   - IN time (first detection)
   - OUT time (after 4+ hours)
- Stores data in MySQL database
- Displays attendance in dashboard
