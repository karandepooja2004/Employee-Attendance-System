# 🎓 Employee Attendance System

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

---

## 📂 Dataset & CSV Creation
### 📸 Dataset Creation

* Run FaceCapture.py file
* Create a folder: dataset/
* Inside it, create subfolders for each student:
  
    dataset/Rahul/

    dataset/Priya/

* Capture 20–50 face images per student using camera
* Images are automatically saved in respective folders

### 📄 CSV Files
1. Person_Info_New.csv

* Stores training data (image paths with labels)

  EX -
  
    Employee_ID,Name,Image_path

    E01,Rahul,dataset/Rahul/0.jpg
  
    E01,Rahul,dataset/Rahul/1.jpg
