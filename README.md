# Shape Detection Using YOLOv8

## Overview
This project implements a simple shape detection system using **YOLOv8**, Python, and OpenCV.  
The model is trained to detect three geometric shapes: **circle**, **square**, and **triangle**, and can perform real-time detection using a webcam.

---

## Workflow

### 1. Dataset Generation  
A synthetic dataset was created using `generate_shape_dataset.py`.  
The script draws random shapes on plain backgrounds and generates YOLO-formatted labels automatically.

### 2. Data Configuration  
A `data.yaml` file specifies:
- Training and validation image paths  
- Number of classes  
- Class names  

### 3. Model Training  
The YOLOv8-nano model (`yolov8n.pt`) was trained using `train_shape_model.py`.  
Training outputs (metrics, graphs, and weights) are stored in `runs/detect/`.

### 4. Real-Time Detection  
Using the trained weights (`best.pt`), `detect_shapes_live.py` performs live shape detection through a webcam.

## Project Structure
generate_shape_dataset.py # Synthetic dataset generator
train_shape_model.py # YOLOv8 training script
detect_shapes_live.py # Real-time detection script
data.yaml # Dataset configuration
results.png # Training metrics and graphs
dataset/ # Generated dataset (train/val)


---

## Technologies Used
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  

---
  
👉## Demo Video
👉 [Click to Watch the Demo](https://drive.google.com/uc?export=download&id=128GiH_i48tXD5yfET6QJRfNV98DbM0N8)




