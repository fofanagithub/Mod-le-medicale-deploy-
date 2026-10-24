# Model card — MedXRay v1.0.0

Format inspiré de Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

## Détails du modèle

| | |
|---|---|
| Tâche | Classification binaire de radiographies thoraciques de face : **normal** / **pneumonie** |
| Architecture | ResNet18 pré-entraîné ImageNet, fine-tuning complet, tête à 1 logit |
| Entrée | Image en niveaux de gris, recadrage carré centré, 224×224, répliquée sur 3 canaux |
| Sortie | P(pneumonie) ∈ [0, 1] et étiquette après application d'un seuil |
| Seuil | Choisi sur la validation : sensibilité ≥ 95 % puis spécificité maximale |
| Explicabilité | Grad-CAM sur la sortie de `layer4` (grille 7×7 interpolée) |
| Code | `src/medxray/` — entraînement : `medxray-train --config configs/train.yaml` |
| Entraînement | 8 époques, CPU, 115 min ; meilleure époque : 7 (AUROC val 0,999) |

## Usage prévu

- **Prévu** : démonstration technique et pédagogique de la mise en production d'un modèle
  d'imagerie (API, conteneur, explicabilité, tests).
- **Hors périmètre** : toute utilisation clinique. Ce modèle n'est **pas un dispositif médical**
  (ni marquage CE au titre du règlement (UE) 2017/745, ni autorisation FDA). Il n'a fait l'objet
  d'aucune validation clinique.

## Données

**PneumoniaMNIST** (MedMNIST v2, Yang et al., *Scientific Data* 2023), version 224×224,
licence CC BY 4.0. Elle dérive du jeu de Kermany et al. (*Cell* 2018) : radiographies
thoraciques **pédiatriques** (enfants de 1 à 5 ans), Guangzhou Women and Children's Medical
Center.

| Split | Images | Prévalence pneumonie |
|---|---:|---:|
| train | 4 708 | 74,2 % |
| val | 524 | 74,2 % |
| test | 624 | 62,5 % |

Les splits officiels sont utilisés tels quels. Le test a une prévalence différente du train :
la VPP/VPN y sont donc spécifiques et ne se transposent pas à une autre population.

## Performances (jeu de test, seuil figé sur la validation)

Voir [`reports/metrics.json`](../reports/metrics.json) (généré par `medxray-evaluate`) pour les valeurs exactes et
les intervalles de confiance à 95 % (bootstrap, 1 000 ré-échantillonnages).

| Métrique | Valeur | IC 95 % |
|---|---:|---:|
| **AUROC** | **0,996** | [0,994 – 0,998] |
| **Sensibilité** (pneumonies détectées) | **97,9 %** (382 / 390) | [96,5 – 99,2] |
| **Spécificité** (normaux reconnus) | **95,3 %** (223 / 234) | [92,3 – 97,7] |
| Exactitude | 97,0 % | [95,5 – 98,2] |
| VPP / VPN | 97,2 % / 96,5 % | — |
| F1 | 0,976 | — |

Seuil de décision : **0,310**, choisi sur la validation (sensibilité ≥ 95 %), puis figé.
Contrôle de validité Grad-CAM : corrélation de Spearman de **0,012** entre les cartes du modèle
entraîné et celles du modèle randomisé. Les explications dépendent donc bien des poids appris.

**Lecture critique.** Ces scores sont élevés, mais ils mesurent une performance *intra-domaine* :
train et test viennent du même hôpital pédiatrique. Ils ne disent rien de la généralisation à
d'autres populations. Les baselines publiées dans le papier MedMNIST v2 sont plus basses, mais
elles ont été obtenues à partir d'images 28×28 sur-échantillonnées : elles ne sont pas
directement comparables.

**Ce que montre Grad-CAM** ([galerie](../reports/gradcam_examples.png)) :
- *Pneumonie* : la carte se concentre sur les champs pulmonaires, surtout les bases, ce qui est
  cohérent avec des opacités.
- *Normal* : la carte se concentre sur le médiastin et la silhouette cardiaque. Le modèle semble
  s'appuyer sur la netteté des contours centraux, ce qui n'est pas la démarche d'un radiologue.
- *Faux positifs* : la carte s'allume parfois en **bordure d'image**. C'est un indice possible
  d'apprentissage par raccourci (cadrage, artefacts), qui serait à investiguer, par exemple avec
  un masquage des bords ou une validation externe.
- Un faux négatif à p ≈ 0,00 suggère aussi du bruit d'annotation, connu sur ce jeu de données.

## Explicabilité : ce que Grad-CAM montre, et ce qu'il ne montre pas

- La carte indique les régions dont les activations **augmentent le score** de la classe
  expliquée. C'est une corrélation, pas un raisonnement causal ni une segmentation de lésion.
- La résolution native est de 7×7 : la localisation est grossière (~32 px par cellule).
- **Contrôle de validité** (Adebayo et al., 2018) : on ré-initialise aléatoirement `layer4` et
  la tête, puis on mesure la corrélation de Spearman entre les cartes avant et après. Une
  corrélation proche de 0 signifie que la carte dépend bien des poids appris et pas seulement
  des contours de l'image. Le résultat est dans `reports/metrics.json` → `gradcam_sanity_check`.

## Limites connues et risques

1. **Population pédiatrique uniquement.** Comportement inconnu sur adultes, autres hôpitaux,
   autres appareils : risque élevé de *domain shift*.
2. **Raccourcis possibles.** Les modèles entraînés sur ce jeu peuvent exploiter des artefacts
   (marqueurs, cadrage, texte incrusté). Grad-CAM permet d'en repérer certains, mais pas tous.
3. **Probabilités non calibrées.** La sortie est un score ordonné. Une calibration (Platt,
   régression isotone) serait nécessaire avant de l'interpréter comme une probabilité clinique.
4. **Pas de rejet hors distribution.** Une photo de chat recevra une prédiction. Pistes :
   détecteur OOD, ou contrôle de cohérence de l'entrée.
5. **Pas de DICOM.** L'API accepte PNG/JPEG. Un déploiement hospitalier lirait du DICOM
   (pydicom), avec fenêtrage et anonymisation des métadonnées.

## Considérations éthiques

- L'API ne conserve aucune image et ne journalise ni le contenu ni le nom des fichiers.
- Un faux négatif peut retarder une prise en charge : c'est pourquoi le seuil privilégie la
  sensibilité, au prix de davantage de faux positifs.
