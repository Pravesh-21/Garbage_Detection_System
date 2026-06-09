# =====================================================================
# LOCAL VS CODE YOLO11 TRAINING ENGINE (WINDOWS OPTIMIZED)
# =====================================================================
import os
import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO('yolo11m.pt')

    # Launch local training profile
    results = model.train(
        data='dataset.yaml',       
        epochs=50,                 
        imgsz=640,                 
        batch=16,                  
        device=0,                  
        workers=2,                 
        cache=False,               
        
        val=True,
        cos_lr=True,               
        patience=15,               
        
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
    best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    if os.path.exists(best_weights_path):
        best_model = YOLO(best_weights_path)
        best_model.export(format='onnx', imgsz=640)
        print(f"\n✅ Success! Download your best.pt and best.onnx models from: {results.save_dir}/weights/")
    else:
        print("\n❌ Error: Could not verify final weights destination file tracks.")