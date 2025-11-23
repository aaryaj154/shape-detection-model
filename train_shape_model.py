from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

# Train the model
results = model.train(
    data='data.yaml',
    epochs=30,        # increase for better accuracy
    imgsz=416,        
    batch=8,          
    name='shapes_detector'
)

print("✅ Training complete! Model saved in: runs/detect/shapes_detector/weights/best.pt")
