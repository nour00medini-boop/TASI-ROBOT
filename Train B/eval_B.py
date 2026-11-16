from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent                 # ...\TASI-ROBOT\Train B
ROOT = HERE.parent                                     # ...\TASI-ROBOT
WEIGHTS = ROOT / "runs" / "detect" / "runs_tasi" / "B_taco_webots" / "weights" / "best.pt"

JEUX = [("test_reel_taco", HERE / "data_test_taco.yaml"),
        ("test_sim_webots", HERE / "data_test_webots.yaml")]

if __name__ == "__main__":
    assert WEIGHTS.exists(), WEIGHTS
    model = YOLO(str(WEIGHTS))
    for nom, yaml in JEUX:
        assert yaml.exists(), yaml
        m = model.val(data=str(yaml), split="test", imgsz=640, device=0,
                      conf=0.001, iou=0.6, plots=True,
                      project=str(ROOT / "runs" / "detect" / "runs_tasi"),
                      name=f"eval_B_{nom}", exist_ok=True)
        print(f"\n=== {nom} ===")
        print(f"mAP@0.5      : {m.box.map50:.3f}")
        print(f"mAP@0.5:0.95 : {m.box.map:.3f}")
        print(f"P (moy)      : {m.box.mp:.3f}")
        print(f"R (moy)      : {m.box.mr:.3f}")
        print("AP50 par classe :", dict(zip(m.names.values(), m.box.ap50.round(3))))
        print("Vitesse (ms)    :", m.speed)