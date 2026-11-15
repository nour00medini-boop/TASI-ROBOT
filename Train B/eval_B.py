from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\nourmedini\Desktop\TASI-ROBOT\runs\detect\runs_tasi\B_taco_webots\weights\best.pt")
    for nom, yaml in [("TEST REEL (TACO)", "data_test_taco.yaml"),
                      ("TEST SIMULATION (Webots)", "data_test_webots.yaml")]:
        m = model.val(data=yaml, split="test", imgsz=640, device=0,
                      conf=0.001, iou=0.6, plots=True,
                      project="runs_tasi", name=f"eval_B_{nom[:8].strip()}", exist_ok=True)
        print(f"\n=== {nom} ===")
        print(f"mAP@0.5      : {m.box.map50:.3f}")
        print(f"mAP@0.5:0.95 : {m.box.map:.3f}")
        print(f"P (moy)      : {m.box.mp:.3f}")
        print(f"R (moy)      : {m.box.mr:.3f}")
        print("AP50 par classe :", dict(zip(m.names.values(), m.box.ap50.round(3))))
        print(f"Vitesse (ms) : {m.speed}")