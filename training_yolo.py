# =====================================================================
# LOCAL VS CODE YOLO11 TRAINING ENGINE (WINDOWS OPTIMIZED)
# =====================================================================
import os
import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    # Critical Windows safety guard for multi-threaded dataloading
    multiprocessing.freeze_support()
    
    # Initialize the YOLO11 Medium weights framework
    # It will automatically download 'yolo11m.pt' to your local folder on run
    model = YOLO('yolo11m.pt')

    # Launch local training profile
    results = model.train(
        data='dataset.yaml',       # Points directly to your local yaml config file
        epochs=50,                 # 50 full epochs for high accuracy feature mapping
        imgsz=640,                 # 640px keeps execution fast on consumer hardware
        batch=16,                  # Safe memory allocation size for local VRAM footprints
        device=0,                  # Uses your primary local NVIDIA GPU (Set to 'cpu' if no discrete graphics)
        workers=2,                 # Kept at 2 to keep Windows responsive while processing backlogs
        cache=False,               # Disables heavy caching to protect system RAM limits
        
        # --- Advanced Accuracy Settings ---
        val=True,
        cos_lr=True,               # Smooth learning rate decay curves for fine late-stage tuning
        patience=15,               # Early stopping threshold if accuracy peaks early
        
        # --- Intense Augmentations for 90%+ Target Tracking ---
        fliplr=0.5,
        mosaic=1.0,
        hsv_v=0.4,
        hsv_s=0.7,
        scale=0.6,
        mixup=0.15,
        copy_paste=0.15,
        degrees=10.0,
        
        project='garbage_detection',
        name='yolo11m_local_run'
    )

    print("\nTraining execution complete. Initiating auto-export sequence...")
    
    # Locate weights dynamically relative to the training project output directory
    best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    if os.path.exists(best_weights_path):
        best_model = YOLO(best_weights_path)
        # Exporting to ONNX format for micro-controller deployment or app integration profiles
        best_model.export(format='onnx', imgsz=640)
        print(f"\n✅ Success! Download your best.pt and best.onnx models from: {results.save_dir}/weights/")
    else:
        print("\n❌ Error: Could not verify final weights destination file tracks.")