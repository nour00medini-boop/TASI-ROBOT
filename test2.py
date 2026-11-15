from ultralytics import YOLO

DATA = r"C:\Users\nourmedini\Desktop\TASI-ROBOT\Webots Robot Collecteur\data_webots.yaml"

if __name__ == "__main__":
    m = YOLO(r"C:\Users\nourmedini\Desktop\TASI-ROBOT\runs\detect\sanity\weights\best.pt")
    m.val(data=DATA, split="val", conf=0.25, iou=0.6, plots=True,
          device=0, name="sanity_conf025", exist_ok=True)
