import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

MODEL_PATH = r"E:\Garbage_Detection_System\runs\detect\garbage_detection\yolo11m_local_run\weights\best.pt"

st.set_page_config(
    page_title="Garbage Detection System",
    page_icon="♻️",
    layout="wide"
)

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception:
        st.warning("Custom weights not found yet. Using base YOLO11m for UI testing.")
        return YOLO("yolo11m.pt")

model = load_model(MODEL_PATH)

def process_frame(frame, conf_threshold):
    """Runs YOLO inference on a single frame and returns the annotated frame and box counts."""
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()
    
    # Extract class counts
    boxes = results[0].boxes
    class_counts = {}
    if len(boxes) > 0:
        detected_classes = [model.names[int(cls)] for cls in boxes.cls]
        class_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}
        
    return annotated_frame, class_counts



def upload_image(conf_threshold):
    """Handles static image uploading, processing, and display."""
    st.subheader("📷 Image Upload Detection")
    uploaded_file = st.file_uploader("Choose a scene image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing image layers..."):
                # Run inference
                annotated_img, counts = process_frame(img_array, conf_threshold)
                # Convert BGR back to RGB for Streamlit display
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, caption="Processed Detection", use_container_width=True)
        
        # Display summary metrics below
        if counts:
            st.success(f"Detected {sum(counts.values())} garbage items!")
            for cls, count in counts.items():
                st.write(f"- **{cls.capitalize()}**: {count}")
        else:
            st.info("No garbage items detected.")


def upload_video(conf_threshold):
    """Handles video file upload, frame-by-frame processing, and playback."""
    st.subheader("🎥 Video Analysis")
    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Streamlit needs a temporary file on disk to pass to OpenCV's VideoCapture
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        vid_cap = cv2.VideoCapture(tfile.name)
        st.info("Processing video stream. Turn off or change mode to cancel.")
        
        # Placeholder window to stream processed frames into
        video_placeholder = st.image([])
        
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret:
                break
            
            # Process individual frame
            annotated_frame, _ = process_frame(frame, conf_threshold)
            # OpenCV captures BGR, stream expects RGB
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
        vid_cap.release()
        st.success("Video processing complete!")


def live_monitoring(conf_threshold):
    """Handles real-time webcam/CCTV hardware camera feeds."""
    st.subheader("🔴 Live Feed Monitoring")
    st.write("Toggle the checkbox below to initialize your system camera hardware.")
    
    run_cam = st.checkbox("Activate Camera Stream")
    cam_placeholder = st.image([])
    
    if run_cam:
        # '0' defaults to built-in webcam. Replace with RTSP URL string for network IP cameras.
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open video device or webcam source.")
            return

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from video source.")
                break
                
            annotated_frame, _ = process_frame(frame, conf_threshold)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            cam_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
        cap.release()
        cam_placeholder.empty()
        st.info("Camera feed terminated securely.")


def main():
    st.title("♻️ Garbage Detection System Dashboard")
    st.write("An automated ecosystem for identifying localized littering and managing municipal waste categories.")
    st.markdown("---")
    
    # Sidebar control vectors
    st.sidebar.header("Control Matrix")
    confidence = st.sidebar.slider("Confidence Limit", 0.0, 1.0, 0.25, 0.05)
    
    mode = st.sidebar.radio(
        "Select Operations Interface",
        ["Static Image Processing", "Batch Video Processing", "Real-Time Surveillance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Active Core Model Architecture: \n`{MODEL_PATH.split('\\')[-1]}`")

    # Routing matrix based on sidebar selection
    if mode == "Static Image Processing":
        upload_image(confidence)
    elif mode == "Batch Video Processing":
        upload_video(confidence)
    elif mode == "Real-Time Surveillance":
        live_monitoring(confidence)

if __name__ == "__main__":
    main()