from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.pt')

results = model.predict(show=True, source='0')

print(results)