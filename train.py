from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,      # GPU
        workers=2      # IMPORTANT for Windows
    )

if __name__ == "__main__":
    main()
    