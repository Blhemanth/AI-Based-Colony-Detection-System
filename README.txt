# 🧫 AI Bacterial Colony Detection System

## 📌 Overview
This project is an AI-based system developed to detect and count bacterial colonies from petri dish images using deep learning.  
It uses the YOLOv8 object detection model to automatically identify colonies and provide an accurate count.

---

## 🚀 Features
- 🔍 Automatic bacterial colony detection  
- 🔢 Accurate colony counting  
- 🎯 Confidence-based filtering to remove false detections  
- 📐 Area filtering to eliminate noise  
- 🖥️ User-friendly GUI for image upload and visualization  
- 💾 Save processed output images  

---

## 🧠 Tech Stack
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Tkinter (GUI)  
- PyTorch  

---

## ⚙️ How It Works
1. User uploads an image through the GUI  
2. The YOLOv8 model detects bacterial colonies  
3. Confidence and area filtering are applied  
4. Colonies are counted automatically  
5. Results are displayed and can be saved  

---

## 🖥️ GUI Preview
(Add screenshots here)

---

## 📂 Project Structure
bacterial-colony-detection/
│
├── app.py # GUI application
├── predict.py # Image prediction script
├── train.py # Model training script
├── runs/ # Training outputs
├── requirements.txt # Dependencies
└── README.md


---

## ▶️ How to Run

### 1. Install dependencies


pip install ultralytics opencv-python pillow


### 2. Run the application

python app.py


---

## 📊 Results
- Successfully detects and counts bacterial colonies  
- Reduces manual effort and human error  
- Works efficiently on GPU-supported systems  

---

## ⚠️ Limitations
- Accuracy depends on dataset quality  
- Overlapping colonies may reduce detection accuracy  
- Performance varies with lighting conditions  

---

## 🔮 Future Improvements
- Real-time webcam detection  
- Colony size measurement  
- Report generation (CSV/PDF)  
- Improved dataset for higher accuracy  

---

## 👨‍💻 Author
Hemanth B L

---

## 📌 Acknowledgment
This project uses the Ultralytics YOLOv8 framework for object detection.