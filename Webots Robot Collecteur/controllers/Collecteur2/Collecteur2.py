#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controllers/collecteur/collecteur.py

Contrôleur de DIAGNOSTIC pour le Pioneer 3-DX — projet TASI-ROBOT (sujet 5).

Objectif : valider la scène Webots (moteurs, caméra, Recognition, lidar, sonars)
SANS importer torch / ultralytics / numpy / scikit-fuzzy.
On isole ainsi tout plantage lié à la chaîne IA.

Mise en place :
  - fichier à placer dans  controllers/collecteur/collecteur.py
  - champ `controller` du Robot dans le .wbt  ->  "collecteur"

Note : ce script ne s'exécute PAS hors de Webots (le module `controller`
n'existe que dans l'environnement Webots).
"""

import sys
import traceback

from controller import Robot

# ===========================================================================
# CONFIGURATION — modifier uniquement ce bloc
# ===========================================================================
TIMESTEP_MS = 32              # pas de simulation (ms)

ENABLE_CAMERA      = True
ENABLE_RECOGNITION = True     # exige un noeud Recognition dans la Camera
ENABLE_LIDAR       = True
ENABLE_SONARS      = True

DRIVE_ENABLED     = True      # False -> robot immobile (test capteurs seuls)
TEST_TORCH_IMPORT = False     # True -> teste "import torch" DANS Webots (peut planter)
LOG_EVERY_N_STEPS = 30        # ~1 s à 32 ms

V_LEFT  = 1.0                 # rad/s — rotation lente sur place pour balayer
V_RIGHT = -1.0

DEV_LEFT   = "left wheel"
DEV_RIGHT  = "right wheel"
DEV_CAMERA = "camera"
DEV_LIDAR  = "lidar"
NB_SONARS  = 16

# ===========================================================================
# Utilitaires
# ===========================================================================
def to_list(seq):
    """Convertit un tableau ctypes (Webots) en liste de floats Python."""
    if seq is None:
        return []
    try:
        return [float(v) for v in seq]
    except TypeError:
        return []

# ===========================================================================
# Contrôleur
# ===========================================================================
class Collecteur:
    def __init__(self):
        self.robot = Robot()
        self.dt = int(self.robot.getBasicTimeStep())
        self.steps = 0
        self.missing = []

        self.left   = self._motor(DEV_LEFT)
        self.right  = self._motor(DEV_RIGHT)
        self.camera = self._camera(DEV_CAMERA)
        self.lidar  = self._lidar(DEV_LIDAR)
        self.sonars = self._sonars()

        self._startup_report()

    # -- récupération défensive des devices --------------------------------
    def _motor(self, name):
        dev = self.robot.getDevice(name)
        if dev is None:
            self.missing.append(name)
            return None
        dev.setPosition(float("inf"))   # mode vitesse
        dev.setVelocity(0.0)
        return dev

    def _camera(self, name):
        cam = self.robot.getDevice(name)
        if cam is None:
            self.missing.append(name)
            return None
        cam.enable(self.dt)
        if ENABLE_RECOGNITION:
            try:
                cam.recognitionEnable(self.dt)
            except Exception as exc:
                print("[WARN] recognitionEnable : %s" % exc, file=sys.stderr)
        return cam

    def _lidar(self, name):
        dev = self.robot.getDevice(name)
        if dev is None:
            self.missing.append(name)
            return None
        dev.enable(self.dt)
        try:
            dev.enablePointCloud()
        except Exception:
            pass
        return dev

    def _sonars(self):
        found = []
        for i in range(NB_SONARS):
            name = "so%d" % i
            dev = self.robot.getDevice(name)
            if dev is None:
                self.missing.append(name)
                found.append(None)
                continue
            dev.enable(self.dt)
            found.append(dev)
        return found

    # -- rapport de démarrage ----------------------------------------------
    def _startup_report(self):
        print("=" * 64)
        print("collecteur.py - controleur de diagnostic")
        print("  python     :", sys.version.split()[0])
        print("  executable :", sys.executable)
        print("  timestep   :", self.dt, "ms")
        if self.missing:
            print("  ABSENTS    :", ", ".join(self.missing))
        else:
            print("  devices    : tous trouves")
        self._report_torch()
        print("=" * 64)
        sys.stdout.flush()

    def _report_torch(self):
        if not TEST_TORCH_IMPORT:
            print("  torch      : non importe (TEST_TORCH_IMPORT = False)")
            return
        try:
            import torch
            print("  torch      : %s | cuda=%s"
                  % (torch.__version__, torch.cuda.is_available()))
        except Exception:
            print("  torch      : IMPORT IMPOSSIBLE", file=sys.stderr)
            traceback.print_exc()

    # -- lecture des capteurs ----------------------------------------------
    def read_recognitions(self):
        """Retourne la liste des objets reconnus (nœud Recognition)."""
        if self.camera is None or not ENABLE_RECOGNITION:
            return []
        try:
            objs = self.camera.getRecognitionObjects()
        except Exception as exc:
            print("[WARN] getRecognitionObjects : %s" % exc, file=sys.stderr)
            return []

        out = []
        for o in objs:
            try:
                out.append({
                    "id":       o.getId(),
                    "model":    o.getModel(),
                    "pos":      to_list(o.getPosition()),
                    "img_pos":  to_list(o.getPositionOnImage()),
                    "img_size": to_list(o.getSizeOnImage()),
                })
            except Exception as exc:
                print("[WARN] objet illisible : %s" % exc, file=sys.stderr)
        return out

    def read_lidar(self):
        """Distance minimale mesurée par le lidar (m), ou None."""
        if self.lidar is None:
            return None
        try:
            ranges = self.lidar.getRangeImage()
        except Exception:
            return None
        if not ranges:
            return None
        return min(float(r) for r in ranges)

    def read_sonars(self):
        """Valeurs brutes des 16 sonars (None si device absent)."""
        return [None if d is None else d.getValue() for d in self.sonars]

    # -- actionneurs --------------------------------------------------------
    def drive(self, left, right):
        if self.left is not None:
            self.left.setVelocity(left)
        if self.right is not None:
            self.right.setVelocity(right)

    # -- boucle principale --------------------------------------------------
    def run(self):
        while self.robot.step(self.dt) != -1:
            self.steps += 1

            if DRIVE_ENABLED:
                self.drive(V_LEFT, V_RIGHT)

            if self.steps % LOG_EVERY_N_STEPS != 0:
                continue

            d_min = self.read_lidar()
            son = [v for v in self.read_sonars() if v is not None]
            recos = self.read_recognitions()

            print("[t=%5.1fs] lidar_min=%s  sonar_min=%s  objets=%d" % (
                self.steps * self.dt / 1000.0,
                "n/a" if d_min is None else "%.2f" % d_min,
                "n/a" if not son else "%.2f" % min(son),
                len(recos),
            ))

            for r in recos:
                print("    model=%-12s id=%-3s img_pos=%s img_size=%s" % (
                    r["model"], r["id"],
                    ["%.0f" % v for v in r["img_pos"]],
                    ["%.0f" % v for v in r["img_size"]],
                ))
            sys.stdout.flush()

def main():
    try:
        Collecteur().run()
    except Exception:
        print("[FATAL] exception dans le controleur :", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
