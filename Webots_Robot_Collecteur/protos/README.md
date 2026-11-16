# PROTOs TASI-ROBOT — étape 2

Cinq PROTOs Webots, un par classe YOLO cible + le leurre.

| Fichier | Classe YOLO | `model` | Rôle |
|---|---|---|---|
| Canette.proto | canette | canette | cible à collecter |
| Bouteille.proto | bouteille | bouteille | cible à collecter |
| PaperCup.proto | paper_cup | paper_cup | cible à collecter |
| Tissues.proto | tissues | tissues | cible à collecter |
| AutreObjet.proto | autre_objet | autre_objet | leurre à NE PAS collecter |

## Installation

1. Copier les 5 `.proto` dans :
`C:\Users\nourmedini\Desktop\TASI-ROBOT\webots\protos\`
2. Ouvrir `collecteur.wbt` dans Webots.
3. `Add a node` > **PROTO nodes (Current Project)** > les 5 apparaissent.
4. Vérifier `model` dans l'inspecteur : il DOIT être exactement le nom de classe (minuscules).

## Points d'attention

- Header `#VRML_SIM R2025a` : à remplacer par `R2024b` (ou autre) si la version de Webots diffère.
- `recognitionColors` est sur le `Shape`, pas sur le `Material` : ne pas déplacer.
- Les tailles sont à l'échelle réelle (canette 12 cm, bouteille 25 cm, etc.).
Si les objets sont trop petits à l'image (img_size < 10 px à 1,5 m),
multiplier `radius`/`height`/`size` par ~1,8 et relancer.

## Ce qui n'est PAS encore fait

- Pas de textures (couleur unie) : suffisant pour valider la chaîne,
insuffisant pour un YOLO robuste. Textures + randomisation → étape 3.
- Une seule variante par classe : `autre_objet` gagnerait à avoir
2–3 variantes (boîte, conserve, gobelet mousse) pour éviter que le modèle
apprenne un motif rigide.