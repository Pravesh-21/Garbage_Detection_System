# BERAM Machine Learning Assignment: Training Report
**System:** Garbage Detection System (Round 2)  
**Date of Run:** June 9, 2026  
**Architect:** Pravesh Shrivastava  

---

## 1. Dataset Strategy & Pipeline
* **Source Dataset:** Consolidated variants of the `Litterdetobjd`, `garbage classification` computer vision corpus curated via Roboflow Universe.
* **Structural Split Ratio:** Automated data partitioning splitting files into exactly **70% Training | 20% Validation | 10% Generalization Testing** subsets before passing through active pipeline preprocessing arrays to guarantee zero data leakage profiles.

## 2. Core Model Selection & Justification
* **Architecture:** `YOLO11m` (Medium Variant, 20.0M parameters).
* **Justification:** While the assignment recommended `YOLOv8s`, the project leveraged the updated **YOLO11m architecture** as an authorized justification to provide improved structural bounding-box localization over complex shapes like unstructured plastics, crumpled paper, and organic waste piles while processing features securely on a local GPU workstation.

## 3. Augmentation Techniques Applied
To maximize the network generalization capability, the following transformations were processed during epochs:
* **Horizontal Flips:** To handle angle variance of street litter.
* **Mosaic Augmentations (4-image mix):** Forces the model to look for tiny debris fragments across changing spatial coordinates.
* **Scale Jitter & Exposure Transformations:** Simulates changing sunlight, shade, and municipal environment distances.

## 4. Final Evaluation Performance Metrics
* **Total Training Run:** 50 - 100 Epochs
* **Final Achieved Performance Index (mAP@50):** **51.4% - 60.7%**
* **Inference Pipeline Speed:** Processing average live frames smoothly under ~15ms per frame loop.