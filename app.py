import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import requests
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"E:\Garbage_Detection_System\runs\detect\garbage_detection\yolo11m_local_run\weights\best.pt"

st.set_page_config(
    page_title="BERAM Garbage Detection System",
    page_icon="♻️",
    layout="wide"
)

# --- INITIALIZE SESSION STATE FOR BONUS FEATURES ---
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []  # Stores history with timestamps 

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception:
        st.warning("Custom trained model weights not found at the specified path. Loading default YOLO11m for UI demo purposes.")
        return YOLO("yolo11m.pt")

model = load_model(MODEL_PATH)

# --- INFERENCE HELPER WITH LOGGING MECHANISM ---
def process_frame(frame, conf_threshold):
    """Runs YOLO inference, annotates the frame, and registers detections to the log."""
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()
    
    boxes = results[0].boxes
    class_counts = {}
    
    if len(boxes) > 0:
        detected_classes = [model.names[int(cls)] for cls in boxes.cls]
        class_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}
        
        # Log the detection with timestamp (Bonus Feature) 
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] Detected: " + ", ".join([f"{k} ({v})" for k, v in class_counts.items()])
        
        # Prevent duplicate log spamming from consecutive video frames
        if not st.session_state.detection_log or st.session_state.detection_log[-1][10:] != log_entry[10:]:
            st.session_state.detection_log.append(log_entry)
            # Keep log concise (last 15 entries)
            if len(st.session_state.detection_log) > 15:
                st.session_state.detection_log.pop(0)
                
    return annotated_frame, class_counts


# --- CORE OPERATIONS FUNCTIONS ---

def upload_image(conf_threshold):
    """Handles static image uploading, processing, and display."""
    st.subheader("📷 Image Upload Detection Interface")
    uploaded_file = st.file_uploader("Choose a scene image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Uploaded Image", use_container_width=True)
            
        with col2:
            with st.spinner("Processing deep network layers..."):
                annotated_img, counts = process_frame(img_array, conf_threshold)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, caption="Processed Detection Output", use_container_width=True)
        
        # UI Metrics Display Layer 
        st.markdown("### 📊 Detection Breakdowns")
        if counts:
            total_items = sum(counts.values())
            st.success(f"Successfully localized {total_items} items of garbage in scene.")
            
            # Display items neatly as structured metric columns [cite: 34, 41]
            metric_cols = st.columns(min(len(counts), 6))
            for i, (cls_name, count) in enumerate(counts.items()):
                with metric_cols[i % 6]:
                    st.metric(label=cls_name.capitalize(), value=count)
        else:
            st.info("No garbage items detected above selected confidence threshold limit.")


def upload_video(conf_threshold):
    """Handles video file upload, frame-by-frame processing, and playback."""
    st.subheader("🎥 Video File Analysis")
    uploaded_video = st.file_uploader("Choose an environment video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        vid_cap = cv2.VideoCapture(tfile.name)
        st.info("Streaming input video buffer. Switch tabs to cancel execution loop.")
        
        video_placeholder = st.image([])
        
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret:
                break
            
            annotated_frame, _ = process_frame(frame, conf_threshold)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
        vid_cap.release()
        st.success("Video data stream parsing completed successfully.")


def live_monitoring(conf_threshold):
    """Handles real-time webcam/CCTV hardware camera feeds."""
    st.subheader("🔴 Live Feed Surveillance Room")
    st.write("Activate the system hardware tracking toggle switch to map active feed matrix windows.")
    
    run_cam = st.checkbox("Activate Real-Time Camera Stream")
    cam_placeholder = st.image([])
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Hardware Alert: Unable to mount local webcam architecture source.")
            return

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Hardware Alert: Loss of signal frame data incoming from channel source.")
                break
                
            annotated_frame, _ = process_frame(frame, conf_threshold)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
        cap.release()
        cam_placeholder.empty()
        st.info("Surveillance camera session closed securely.")


# --- MAIN ENGINE CONTROLLER ---
def main():
    st.title("♻️ Intelligent Garbage Detection System")
    st.write("BERAM Round 2 Automated Municipal Surveillance & Object Localization Dashboard.") [cite: 4]
    st.markdown("---")
    
    # --- SIDEBAR CONTROL VECTOR PANEL ---
    st.sidebar.header("🕹️ System Control Matrix")
    confidence = st.sidebar.slider("Confidence Parameter Limit", 0.0, 1.0, 0.35, 0.05)
    
    mode = st.sidebar.radio(
        "Select Operations Interface",
        ["Static Image Processing", "Batch Video Processing", "Real-Time Surveillance"] [cite: 34]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"🧬 Engine Weights Block: \n`{MODEL_PATH.split('\\')[-1]}`")

    # --- FASTAPI BACKEND MONITOR BLOCK [cite: 34] ---
    st.sidebar.markdown("### 🌐 Backend Gateway Stream")
    try:
        api_check = requests.get("http://127.0.0.1:8000/", timeout=1)
        if api_check.status_code == 200:
            st.sidebar.success("🟢 API Gateway Route Active (/predict)") [cite: 34]
    except requests.exceptions.ConnectionError:
        st.sidebar.warning("⚠️ API Gateway Offline (Run 'python api.py')")

    # --- HISTORICAL DETECTION LOG PANEL (Bonus Feature)  ---
    st.sidebar.markdown("### 📜 Real-time Detection Log") [cite: 42]
    if st.sidebar.button("Clear Log History"):
        st.session_state.detection_log = []
        
    if st.session_state.detection_log:
        for log in reversed(st.session_state.detection_log):
            st.sidebar.caption(log)
    else:
        st.sidebar.text("No active detections logged.")

    # --- SYSTEM INTERFACE ROUTING ---
    if mode == "Static Image Processing":
        upload_image(confidence)
    elif mode == "Batch Video Processing":
        upload_video(confidence)
    elif mode == "Real-Time Surveillance":
        live_monitoring(confidence)

if __name__ == "__main__":
    main()