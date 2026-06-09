from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI(title="BERAM Garbage Detection API System")

# Load your custom fine-tuned weights
MODEL_PATH = r"E:\Garbage_Detection_System\runs\detect\garbage_detection\yolo11m_local_run\weights\best.pt"
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "Online", "system": "Garbage Detection API Gateway"}

@app.post("/predict")
async def predict_garbage(file: UploadFile = File(...)):
    # Read raw image bytes incoming from endpoint payload
    request_bytes = await file.read()
    image = Image.open(io.BytesIO(request_bytes)).convert("RGB")
    
    # Convert PIL structure to NumPy OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Run Inference
    results = model.predict(source=open_cv_image, conf=0.25, verbose=False)
    boxes = results[0].boxes
    
    detections = []
    for box in boxes:
        # Extract individual coordinates and prediction properties
        xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        
        detections.append({
            "class_label": label,
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "bounding_box": {
                "xmin": round(xyxy[0], 2),
                "ymin": round(xyxy[1], 2),
                "xmax": round(xyxy[2], 2),
                "ymax": round(xyxy[3], 2)
            }
        })
        
    return {
        "total_detections_found": len(detections),
        "detections": detections
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)