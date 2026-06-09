# ♻️ Intelligent Garbage Detection System

An automated municipal surveillance, waste management, and object localization ecosystem utilizing a fine-tuned YOLO11m architecture. The system features a public real-time Streamlit dashboard alongside a high-performance FastAPI backend gateway for model inference.
🔗 **Live Production Link:** [https://garbagedetectionsystem.streamlit.app/](https://garbagedetectionsystem.streamlit.app/)
💾 **Download Dataset:** [https://www.kaggle.com/datasets/pravesh212006/garbage-detection-dataset](https://www.kaggle.com/datasets/pravesh212006/garbage-detection-dataset)
---

## 📊 Performance Benchmarks

* **mAP@50 (Overall):** **77.2%** (Exceeds the strong submission target of 70%+)
* **Framework:** Ultralytics YOLO11m
* **Training Platform:** Free T4 GPU Environment
* **Optimization Strategy:** Hyperparameter tuning over 50 epochs, utilizing early stopping validation metrics to prevent overfitting.

---

## 🧬 Dataset Strategy & Augmentation

To ensure robustness against variations in lighting, background clutter, and camera angles, an intelligent data pipeline was constructed by combining municipal waste imagery. 

### Data Splits
* **Training Set:** 70%
* **Validation Set:** 20%
* **Testing Set:** 10%
*(Ensured zero data leakage prior to preprocessing operations)*

### Applied Augmentations
To improve object localization on unseen real-world environments, the following transformations were systematically applied during training:
* **Horizontal Flips:** To generalize positional symmetry.
* **Mosaic Augmentation (4-frame):** To enforce multi-scale detection across distant and crowded objects.
* **Brightness & Contrast Jittering:** To simulate multi-time-of-day surveillance captures.
* **Scale Jittering:** To protect the network from bounding box resolution dependencies.

---

## 🕹️ Must-Have & Bonus Features Implemented

* **Static Image Upload Interface:** Generates precise bounding boxes, dynamic class labels, and localized item counts.
* **Batch Video Processing:** Iterates frame-by-frame through uploaded environmental videos.
* **Real-Time Surveillance Room:** Mounts local camera hardware feeds securely for live tracking.
* **FastAPI Backend Stream:** A robust `/predict` endpoint exposing a scalable JSON API layer.
* **Historical Detection Log (Bonus):** Maintains a sliding timestamp log of consecutive localized frames to prevent duplicate alert spamming.
* **Structured UI Metrics (Bonus):** Neatly organizes real-time localized breakdowns into dynamic, digestible column widgets.

---

## 
🛠️ Local Installation & Setup

Follow these steps to run the complete pipeline locally on your machine.

### 1. Clone the Repository
git clone [https://github.com/Pravesh-21/Garbage_Detection_System.git](https://github.com/Pravesh-21/Garbage_Detection_System.git)
cd Garbage_Detection_System

### 2. Set Up a Virtual Environment
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Architecture Engines

**To launch the FastAPI Backend Gateway (`http://127.0.0.1:8000`):***
python api.py

##📂 Repository Artifacts Matrix
app.py - Core Streamlit frontend application layer.  
api.py - FastAPI gateway endpoints.  
best.pt - Custom fine-tuned model weights mandatory for deployment.  
requirements.txt - Python package dependencies.  
training_report.md - Full documentation covering metrics, dataset choices, and architectural insights.  
results.png & confusion_matrix.png - Automatically generated training evaluation artifacts proving authentic model convergence.