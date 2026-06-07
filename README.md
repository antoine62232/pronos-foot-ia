# ⚽ Prediktora — Prédictions Coupe du Monde 2026

> Une IA qui prédit la Coupe du Monde 2026, de la phase de groupes au vainqueur final, à partir d'un classement Elo maison et de la forme récente des équipes.

[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://prediktora.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏆 Le verdict de l'IA

En simulant l'ensemble du tournoi, l'application désigne son champion prédit pour 2026 :

- **🥇 Champion prédit : Portugal**

Le détail complet (classements des 12 groupes, 32 qualifiés, et l'intégralité du bracket des matchs à élimination directe) est consultable dans l'application.

---

## 🚀 Démo en ligne

**[Tester l'application](https://prediktora.streamlit.app/)**

---

## 🎯 Fonctionnalités

- **Phase de groupes** : prédiction des 72 matchs, classement des 12 groupes, et identification des 32 qualifiés (24 directs + 8 meilleurs troisièmes).
- **Phase à élimination directe** : simulation du bracket complet (1/16 jusqu'à la finale), avec un départage automatique en cas de match nul prédit.
- **Réalité VS IA** : pendant le tournoi, chaque pronostic de l'IA est confronté au vrai résultat, avec un taux de réussite mis à jour automatiquement (voir « Résultats en direct »).
- **Interface** : 5 onglets (les matchs, mes pronos, l'IA explique, phase éliminatoire, Réalité VS IA), un système de pronostics utilisateur pour se comparer à l'IA, et des probabilités affichées pour chaque match.

---

## 🧠 Comment ça marche

Le modèle est un classifieur **XGBoost** qui prédit l'issue d'un match en trois classes : victoire à domicile (1), match nul (N) ou victoire à l'extérieur (2).

Il s'appuie sur **11 variables** :

- **La forme récente** de chaque équipe : moyenne de buts marqués et encaissés sur ses 5 et ses 10 derniers matchs (forme à domicile pour l'équipe qui reçoit, à l'extérieur pour la visiteuse).
- **Un classement Elo** calculé maison sur tout l'historique international. Chaque équipe possède un score de force qui s'ajuste après chaque match selon la surprise du résultat : battre une équipe plus forte rapporte plus de points qu'en battre une plus faible. Ce classement évolue dans le temps et couvre toutes les équipes.
- **Le contexte** : match sur terrain neutre ou non.

Le classement Elo a remplacé un précédent indicateur de force figé (les points FIFA), ce qui a amélioré les prédictions tout en supprimant tout risque de « fuite de données » au backtest.

---

## 🤖 Résultats en direct

Pendant la compétition, l'application ne se contente pas de prédire : elle se confronte à la réalité. Un **robot GitHub Actions** s'exécute automatiquement toutes les deux heures, interroge l'API **football-data.org** pour récupérer les scores des matchs terminés, et enregistre le tout dans `data/resultats_reels.csv` (committé directement dans le dépôt).

L'application lit ensuite ce fichier — et non l'API en direct. Ce choix d'architecture a trois avantages :

- **Quota préservé** : quel que soit le nombre de visiteurs, ils lisent un fichier ; aucun appel API n'est déclenché côté application.
- **Indépendant de la mise en veille** : les résultats restent à jour même quand l'application dort, puisque le robot tourne sur les serveurs de GitHub.
- **Historique versionné** : chaque mise à jour est un commit, ce qui conserve une trace datée des résultats.

L'onglet **Réalité VS IA** affiche alors, en direct, le pourcentage de pronostics corrects de l'IA ainsi que le détail match par match.

---

## 📊 Performance

Le modèle est évalué par un **backtest à séparation temporelle stricte** : il apprend uniquement sur les matchs antérieurs à 2018, puis prédit **5 822 matchs de compétition** joués ensuite (qualifications, Euro, Copa América, Ligue des Nations, CAN, Coupe du Monde). Aucune information du futur n'entre dans l'entraînement.

L'évaluation ne se limite pas à l'accuracy : elle inclut le **Brier score**, la **log loss** et le **RPS**, qui mesurent la qualité des probabilités (et pas seulement le pronostic final).

| Type de matchs                     | Nombre | Accuracy   | Brier |
| ---------------------------------- | ------ | ---------- | ----- |
| Phases finales (matchs équilibrés) | 2 741  | **53,2 %** | 0,579 |
| Qualifications                     | 3 081  | 61,6 %     | 0,489 |
| **Ensemble des compétitions**      | 5 822  | **57,7 %** | 0,531 |

Repères : un pronostic au hasard donnerait environ 33 % d'accuracy et un Brier de 0,667.

Le chiffre le plus représentatif pour la Coupe du Monde est celui des **phases finales (53,2 %)** : ce sont des matchs équilibrés entre nations de niveau proche, sur terrain neutre, donc difficiles à prédire. Les qualifications, souvent déséquilibrées, gonflent mécaniquement la moyenne.

**Calibration** : les probabilités sont fiables. Quand le modèle annonce 75 % de confiance, l'issue se réalise environ 76 % du temps ; à 89 %, environ 90 %. Le modèle est même légèrement prudent (il gagne un peu plus souvent qu'il ne l'annonce), ce qui est une bonne propriété pour un produit.

---

## 📈 Évolution du modèle

| Modèle               | Force d'équipe                              | Accuracy (phases finales) |
| -------------------- | ------------------------------------------- | ------------------------- |
| Version initiale     | Points FIFA (figés)                         | 51,3 %                    |
| **Version actuelle** | **Classement Elo (variable dans le temps)** | **53,2 %**                |

La démarche : estimer la performance honnêtement via le backtest, comparer plusieurs jeux de variables, puis ne retenir que ce qui apporte un gain réel et mesuré.

---

## 🛠️ Stack technique

| Catégorie        | Outil             | Rôle                                |
| ---------------- | ----------------- | ----------------------------------- |
| Langage          | Python 3.11       | Backend                             |
| Manipulation     | Pandas            | DataFrames                          |
| Calculs          | NumPy             | Opérations vectorielles             |
| Machine Learning | XGBoost           | Classification 1/N/2                |
| Utilitaires ML   | Scikit-learn      | Encodage, pondération des classes   |
| Interface        | Streamlit         | Application web                     |
| Résultats réels  | football-data.org | Récupération des scores du tournoi  |
| Automatisation   | GitHub Actions    | Mise à jour planifiée des résultats |
| Hébergement      | Streamlit Cloud   | Mise en ligne                       |
| Versioning       | Git / GitHub      | Code source                         |

---

## 🏗️ Architecture

```
pronos-foot-ia/
├── data/                              # Donnees et sorties
│   ├── matchs_internationaux.csv      # Historique brut
│   ├── matchs_entrainement.csv        # Matchs d'entrainement (depuis 2000)
│   ├── matchs_a_predire.csv           # 72 matchs de la CdM 2026
│   ├── elo_actuel.csv                 # Classement Elo courant par equipe
│   ├── predictions_groupes.csv        # Predictions de la phase de groupes
│   ├── qualifies_1_16.csv             # 32 qualifies
│   ├── bracket_complet.csv            # Bracket d'elimination directe
│   ├── vainqueur_final.csv            # Champion predit
│   └── resultats_reels.csv            # Resultats reels du tournoi (rempli par le robot)
│
├── src/                               # Scripts Python
│   ├── elo.py                         # Classement Elo (calcul + classement courant)
│   ├── features.py                    # Creation des variables de forme
│   ├── modele.py                      # Entrainement du modele de production
│   ├── metriques.py                   # Metriques d'evaluation (accuracy, log loss, Brier, RPS)
│   ├── backtest_complet.py            # Backtest officiel sur les matchs de competition
│   ├── comparer_features.py           # Comparaison de jeux de variables
│   ├── predictions_groupes.py         # Prediction des classements de groupes
│   └── predictions_phase_eliminatoire.py  # Simulation du bracket
│
├── components/                        # Interface Streamlit (dont l'onglet "Realite VS IA")
├── scripts/
│   └── maj_resultats.py               # Robot : recupere les scores via football-data.org
├── .github/
│   └── workflows/
│       └── maj-resultats.yml          # Planification du robot (GitHub Actions)
├── models/                            # Modele entraine (.pkl)
├── app.py                             # Point d'entree de l'application
├── requirements.txt
└── LICENSE
```

---

## 🚀 Installation et lancement

### Prérequis
- Python 3.11+
- Git

### Installation

```bash
# Cloner le projet
git clone https://github.com/antoine62232/pronos-foot-ia.git
cd pronos-foot-ia

# Creer un environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# Installer les dependances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### Régénérer le modèle et les prédictions

```bash
# 1. Entrainer le modele et produire le classement Elo courant
python src/modele.py

# 2. Predire les classements de groupes et les qualifies
python src/predictions_groupes.py

# 3. Simuler le bracket d'elimination directe
python src/predictions_phase_eliminatoire.py

# (Optionnel) Evaluer le modele sur les matchs de competition
python src/backtest_complet.py
```

---

## 🗺️ Roadmap

### ✅ Réalisé
- Modèle XGBoost de classification 1/N/2.
- Backtest fiable sur 5 822 matchs avec métriques probabilistes et analyse de calibration.
- Classement Elo maison, adopté en production à la place des points FIFA.
- Application déployée, prédisant le tournoi complet de bout en bout.
- Mise à jour automatique des résultats réels pendant le tournoi (robot GitHub Actions + football-data.org) et onglet « Réalité VS IA » de suivi en direct.

### 🔄 En cours / à venir
- Enrichir le modèle avec de nouvelles sources de données (cotes des bookmakers, valeur marchande des effectifs).
- Affiner le classement Elo (différence de buts, importance du match).
- Travail UX/UI et préparation d'une version commercialisable.

---

## 👨‍💻 À propos

Projet personnel mené en autonomie pour pratiquer le Machine Learning sur un sujet concret avec une échéance réelle (la Coupe du Monde 2026). Premier contact avec la data science lors d'un stage chez Stellantis ; ce projet sert à approfondir Python, le ML et le déploiement.

Ce qui m'intéresse : observer un modèle apprendre, s'ajuster et gagner en précision au fil des itérations, avec une évaluation honnête à chaque étape.

---

## 📄 Licence

MIT. Voir le fichier `LICENSE`.

---

## 🙏 Remerciements

- **football-data.co.uk** et les jeux de données publics de résultats internationaux pour l'historique des matchs.
- **football-data.org** pour les scores en direct pendant la compétition.
- **Streamlit Cloud** pour l'hébergement.
- La communauté Python et Data Science pour les outils open source.