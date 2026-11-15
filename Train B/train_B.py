from ultralytics import YOLO

DATA = r"C:\Users\nourmedini\Desktop\TASI-ROBOT\data_B.yaml"


if __name__ == "__main__":       
    model = YOLO("yolov8n.pt")
    model.train(
        data="data_B.yaml",
        epochs=80,
        patience=20,                # arrêt si pas d'amélioration
        imgsz=640,
        batch=16,                   # si "CUDA out of memory" -> 8
        device=0,
        workers=4,                  # si erreurs / blocages -> 0
        cache="ram",                # images en RAM 
        seed=42,
        deterministic=True,
        project="runs_tasi",
        name="B_taco_webots",
        exist_ok=False,
    )