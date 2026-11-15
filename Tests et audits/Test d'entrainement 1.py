from ultralytics import YOLO

DATA = r"C:\Users\nourmedini\Desktop\TASI-ROBOT\Webots Robot Collecteur\data_webots.yaml"

if __name__ == "__main__":
    m = YOLO("yolov8n.pt")
    m.train(data=DATA, imgsz=640, epochs=30, batch=4, device=0, workers=2,
            seed=0, name="sanity", exist_ok=True)