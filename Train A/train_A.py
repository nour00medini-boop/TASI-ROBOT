from pathlib import Path
HERE = Path(__file__).resolve().parent
from ultralytics import YOLO

if __name__ == "__main__":       
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(HERE / "data_A.yaml"),
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
        name="A_taco_seul",
        exist_ok=False,
    )