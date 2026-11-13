#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controllers/capture/capture.py
Génère un dataset YOLO automatiquement à partir de la scène Webots.
"""

import os
import random
import sys

import cv2
import numpy as np
from controller import Supervisor

# ===========================================================================
# CONFIGURATION
# ===========================================================================
OUTPUT_DIR = r"C:\Users\nourmedini\Desktop\TASI-ROBOT\dataset_webots"
CLASSES = {
    "canette": 0,
    "bouteille": 1,
    "paper_cup": 2,
    "tissues": 3,
    "autre_objet": 4
}
NB_IMAGES = 500
TIMESTEP_MS = 32
SETTLE_STEPS = 5

def create_output_dirs():
    img_dir = os.path.join(OUTPUT_DIR, "images")
    lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir

class DatasetCapturer:
    def __init__(self):
        self.robot = Supervisor() 
        self.dt = int(self.robot.getBasicTimeStep())
        
        self.camera = self.robot.getDevice("camera")
        if self.camera is None:
            raise RuntimeError("Camera introuvable. Vérifie le nom dans le .wbt.")
            
        self.camera.enable(self.dt)
        self.camera.recognitionEnable(self.dt)
        
        # --- CORRECTION ICI : Récupération dynamique de la taille ---
        self.cam_width = self.camera.getWidth()
        self.cam_height = self.camera.getHeight()
        print(f"[INFO] Résolution de la caméra détectée : {self.cam_width}x{self.cam_height}")
        
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            raise RuntimeError("getSelf() a échoué. As-tu bien coché 'supervisor' à TRUE dans le .wbt ?")

        self.img_dir, self.lbl_dir = create_output_dirs()
        print(f"[INFO] Sauvegarde dans : {OUTPUT_DIR}")
        print(f"[INFO] Supervisor actif, capture de {NB_IMAGES} images.")

    def teleport_random(self):
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(-1.5, 1.5)
        z = 0.05  
        angle = random.uniform(0, 2 * np.pi)
        
        translation_field = self.robot_node.getField("translation")
        rotation_field = self.robot_node.getField("rotation")
        
        translation_field.setSFVec3f([x, y, z])
        rotation_field.setSFRotation([0, 0, 1, angle])
        
        for _ in range(SETTLE_STEPS):
            self.robot.step(self.dt)

    def process_and_save(self, idx):
        raw_img = self.camera.getImage()
        if raw_img is None:
            return

        # Utilisation des dimensions réelles pour le reshape
        img = np.frombuffer(raw_img, np.uint8).reshape((self.cam_height, self.cam_width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        recos = self.camera.getRecognitionObjects()
        label_lines = []

        for obj in recos:
            model = obj.getModel()
            if model not in CLASSES:
                continue
            
            cid = CLASSES[model]
            pos = obj.getPositionOnImage()
            size = obj.getSizeOnImage()
            
            cx, cy = pos[0], pos[1]
            w, h = size[0], size[1]
            
            # Normalisation YOLO avec les bonnes dimensions
            cx_norm = max(0.0, min(1.0, cx / self.cam_width))
            cy_norm = max(0.0, min(1.0, cy / self.cam_height))
            w_norm = max(0.0, min(1.0, w / self.cam_width))
            h_norm = max(0.0, min(1.0, h / self.cam_height))
            
            # On ignore les objets trop petits (< 10 px)
            if w < 10 or h < 10:
                continue

            label_lines.append(f"{cid} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        filename = f"webots_{idx:04d}"
        img_path = os.path.join(self.img_dir, filename + ".jpg")
        lbl_path = os.path.join(self.lbl_dir, filename + ".txt")

        cv2.imwrite(img_path, img_bgr)
        with open(lbl_path, "w") as f:
            f.write("\n".join(label_lines))

    def run(self):
        for i in range(NB_IMAGES):
            self.teleport_random()
            self.process_and_save(i)
            if (i + 1) % 50 == 0:
                print(f"  Progression : {i + 1}/{NB_IMAGES} images capturées.")
                sys.stdout.flush()
        print("[SUCCÈS] Capture terminée.")

if __name__ == "__main__":
    try:
        DatasetCapturer().run()
    except Exception as e:
        print(f"[ERREUR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
