# ⚽ Pronos Foot IA — Prédictions Coupe du Monde 2026

> **Une IA qui prédit les 72 matchs de la Coupe du Monde 2026 avec une précision proche des bookmakers professionnels.**

[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://pronos-foot-ia.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Démo en ligne

L'application est accessible 24/7 ici : **[pronos-foot-ia.streamlit.app](https://pronos-foot-ia.streamlit.app/)**

---

## 📸 Aperçu

L'application propose 3 onglets :

- **Les matchs** : les pronostics IA pour les 72 matchs de la CdM 2026
- **Mes pronos** : interface permettant à l'utilisateur de pronostiquer chaque match et de se comparer à l'IA
- **L'IA explique** : transparence totale sur le fonctionnement du modèle (performance, importance des features)

---

## 🎯 Objectifs du projet

### Objectif principal

Construire une application web prédictive basée sur le Machine Learning pour la Coupe du Monde 2026, avec :

- Une **précision supérieure au hasard** (33%) et **proche des bookmakers professionnels** (52-55%)
- Une **interface utilisateur claire** accessible aux non-techniciens
- Une **transparence sur le fonctionnement** du modèle (pas de boîte noire)

### Objectifs secondaires

- Pratiquer le **Machine Learning en autonomie** sur un sujet concret
- Apprendre le **déploiement d'une application web** Python
- Documenter le projet pour **partage GitHub** et **communication LinkedIn**
- Mettre en place un **workflow data science professionnel** (backtest, validation croisée, feature engineering)

---

## 🛠️ Stack technique

Chaque outil a été choisi pour une raison précise.

### Langage et manipulation de données

| Outil | Rôle | Pourquoi |
|---|---|---|
| **Python 3.11+** | Langage principal | Standard de fait en Data Science |
| **Pandas** | Manipulation des tableaux de données | Indispensable pour les CSV de 25 000 lignes |
| **NumPy** | Calculs mathématiques | Fondation de l'écosystème scientifique Python |

### Machine Learning

| Outil | Rôle | Pourquoi |
|---|---|---|
| **XGBoost** | Algorithme de classification | Le meilleur algorithme sur données tabulaires |
| **Scikit-learn** | Évaluation, encodage, rééquilibrage | Boîte à outils standard du ML |

### Interface web

| Outil | Rôle | Pourquoi |
|---|---|---|
| **Streamlit** | Création de l'interface web | Permet de faire du web sans HTML/CSS |
| **Plotly** | Graphiques interactifs | Pour les visualisations dynamiques |

### Déploiement et versioning

| Outil | Rôle | Pourquoi |
|---|---|---|
| **Streamlit Cloud** | Hébergement gratuit de l'app | Déploiement en 1 clic depuis GitHub |
| **Git / GitHub** | Versioning et sauvegarde du code | Standard absolu pour tout projet de dev |
| **UptimeRobot** | Maintenir l'app éveillée 24/7 | Évite le sleep mode de Streamlit Cloud |

---

## 🏗️ Architecture du projet
pronos-foot-ia/
│
├── app.py                          # Point d'entrée Streamlit
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
├── components/                     # Interface modulaire
│   ├── styles.py                   # Thème CSS personnalisé
│   ├── data_loader.py              # Chargement des données
│   ├── predictions.py              # Génération des pronostics
│   ├── match_card.py               # Affichage d'un match
│   ├── user_pronos.py              # Système de pronostics utilisateur
│   └── ia_explain.py               # Onglet de transparence du modèle
│
├── data/                           # Données du projet
│   ├── matchs_internationaux.csv   # Dataset brut (25 196 matchs depuis 2000)
│   ├── matchs_entrainement.csv     # Données d'entraînement nettoyées
│   ├── matchs_a_predire.csv        # 72 matchs de la CdM 2026
│   └── backtest_*.csv              # Résultats des backtests
│
├── src/                            # Scripts data science
│   ├── collecte.py                 # Collecte des données
│   ├── exploration.py              # Analyse exploratoire (EDA)
│   ├── nettoyage.py                # Nettoyage des données
│   ├── features.py                 # Création des features
│   ├── modele.py                   # Entraînement du modèle
│   ├── backtest.py                 # Backtest V2 (production)
│   ├── backtest_v3.py              # Expérimentation V3
│   ├── backtest_v4.py              # Expérimentation V4 (feature selection)
│   ├── backtest_v5.py              # Expérimentation V5 (contexte du match)
│   ├── analyse_features.py         # Importance des features
│   └── grid_search.py              # Optimisation des hyperparamètres
│
└── models/                         # Modèles entraînés (sérialisés)
├── modele_football.pkl         # Modèle XGBoost V2
└── label_encoder.pkl           # Encodeur des résultats

### Pourquoi cette structure ?

- **`components/`** : code de l'interface modularisé (un fichier = un composant)
- **`src/`** : scripts data science séparés de l'app pour pouvoir les exécuter individuellement
- **`data/`** : toutes les données au même endroit, du brut au final
- **`models/`** : les modèles entraînés sont versionnés pour pouvoir les charger sans réentraîner

---

## 🔬 Méthodologie

### Cycle de développement Data Science

Le projet suit un cycle ML rigoureux :
Collecte → Exploration → Nettoyage → Feature Engineering → Modélisation → Backtesting → Déploiement

### Backtesting temporellement correct

Pour éviter le **data leakage temporel** (le modèle "voit" le futur pour prédire le passé), le backtesting respecte strictement l'ordre du temps :
Entraînement : matchs AVANT le 20 novembre 2022 (~21 000 matchs)
↓
Test : les 64 matchs réels de la CdM 2022

Cette méthode donne la **vraie performance** du modèle en conditions réelles, comme si on l'avait lancé en novembre 2022.

### Features actuelles (V2)

Le modèle utilise **11 features** :

**Forme des équipes (8 features)**
- Buts marqués/encaissés sur les 5 derniers matchs (domicile et extérieur)
- Buts marqués/encaissés sur les 10 derniers matchs (domicile et extérieur)

**Niveau objectif (2 features)**
- Points FIFA de l'équipe domicile
- Points FIFA de l'équipe extérieur

**Contexte du match (1 feature)**
- Terrain neutre (1 = oui, 0 = non)

### Gestion du déséquilibre des classes

Le dataset est déséquilibré : ~50% de victoires domicile, ~25% de matchs nuls, ~25% de victoires extérieur. Pour éviter que le modèle "ignore" les matchs nuls, on utilise `compute_sample_weight` de Scikit-learn pour **rééquilibrer** automatiquement les poids pendant l'entraînement.

---

## 📊 Performance du modèle

### Backtesting V2 (modèle de production)

Le modèle a été testé sur les Coupes du Monde 2018 et 2022 avec une séparation temporelle stricte :

| Coupe du Monde | Précision | Détail |
|----------------|-----------|--------|
| CdM 2018 | **51.6%** | 33 matchs sur 64 |
| CdM 2022 | **57.8%** | 37 matchs sur 64 |
| **Moyenne** | **54.7%** | **70 matchs sur 128** |

### À titre de comparaison

| Source | Précision |
|---|---|
| Pur hasard | 33% |
| Modèle Pronos Foot IA (V2) | **54.7%** |
| Bookmakers professionnels | 52-55% |
| Goldman Sachs (CdM 2022) | ~55% |

### Cross-Validation (vraie performance)

En cross-validation temporelle (3 splits), le modèle atteint **~49% de précision**. Cette métrique est **plus stricte** que le backtest car elle mesure la performance sur des matchs variés (pas uniquement des CdM).

### Limites identifiées

Les 5 plus grosses erreurs du modèle sur la CdM 2022 sont **les 5 plus grosses surprises du tournoi** :

- Argentine 1-2 Arabie Saoudite
- Cameroun 1-0 Brésil
- Japon 2-1 Espagne
- Tunisie 1-0 France
- Pays-Bas 2-2 Argentine

**Le modèle se trompe là où les humains se trompent : sur l'imprévisible.**

---

## 📈 Évolution du projet (V1 → V5)

Le projet a connu plusieurs itérations, chacune testant une hypothèse de feature engineering.

| Version | Features | CdM 2018 | CdM 2022 | Moyenne | Verdict |
|---------|----------|----------|----------|---------|---------|
| V1 | 7 (forme 5 matchs + FIFA + neutre) | 43.8% | 60.9% | 52.3% | Baseline |
| **V2** | **11 (+ forme 10 matchs)** | **51.6%** | **57.8%** | **54.7%** | ⭐ **Production** |
| V3 | 21 (+ H2H + streaks + clean sheets + confédérations) | 53.1% | 54.7% | 53.9% | Trop de bruit |
| V4 | 14 (feature selection sur V3) | 48.4% | 56.2% | 52.3% | Régression |
| V5 | 16 (+ contexte du match : repos, premier match, tournoi majeur) | 53.1% | 56.2% | 54.7% | Égalité avec V2 |

### Leçons tirées

1. **Le feature engineering compte plus que l'algorithme** : passer de 7 à 11 features (V1 → V2) a fait gagner 2.4 points. Changer XGBoost pour un autre algo n'apporterait que ~0.5 point.

2. **More features ≠ better model** : V3 avec 21 features a régressé. Ajouter des features redondantes ou bruitées dégrade le modèle.

3. **Le sur-apprentissage est le pire ennemi** : sans rééquilibrage, le modèle ignorerait les matchs nuls pour "tricher" sur sa précision.

4. **Le plateau des données internes est atteint** : sans données externes (cotes bookmakers, stats détaillées), il est très difficile de dépasser 55% de précision.

---

## 🚀 Installation et lancement local

### Prérequis

- Python 3.11 ou supérieur
- Git
- Un terminal (CMD, PowerShell, Bash, etc.)

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/antoine62232/pronos-foot-ia.git
cd pronos-foot-ia
```

### Étape 2 — Créer un environnement virtuel

```bash
python -m venv venv
```

Activer l'environnement :

```bash
# Sur Windows
venv\Scripts\activate

# Sur Mac/Linux
source venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 — Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur sur `http://localhost:8501`.

### Étape 5 — Lancer un backtest (optionnel)

```bash
python src/backtest.py
```

---

## 🗺️ Roadmap

### ✅ Phase 1 — V1 publique (terminée)

- App fonctionnelle déployée
- Modèle V2 à 54.7% en backtest
- Premier post LinkedIn

### 🔄 Phase 2 — V2 en cours

- **Session 1 (✅)** : Feature engineering itératif (testé V3, V4, V5)
- **Session 2 (✅)** : Contexte du match (jours de repos, premier match)
- **Session 3 (🔄)** : Intégration des cotes des bookmakers (en cours)

### ⏳ Phase 3 — V3 et au-delà

- Refonte visuelle complète de l'interface
- Possibilité de pronostiquer la phase à élimination directe
- Système de classement utilisateurs (compétition entre joueurs)
- Notifications par email pour les rappels de pronostics
- Version commercialisable pour la Coupe du Monde 2026

---

## 👨‍💻 À propos

Ce projet a été développé en autonomie pour pratiquer le Machine Learning sur un sujet concret avec une vraie deadline (la Coupe du Monde 2026).

J'ai eu une première expérience en data science lors de mon stage chez Stellantis. Ce projet personnel me permet d'approfondir mes compétences en Python, ML et déploiement d'applications web.

**Ce qui m'intéresse vraiment** : observer une machine apprendre, s'ajuster, et gagner en précision au fil des itérations.

### Me contacter

- **LinkedIn** : [Antoine](https://www.linkedin.com/) *(à compléter avec ton URL)*
- **GitHub** : [@antoine62232](https://github.com/antoine62232)

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **football-data.co.uk** pour les données historiques de matchs
- **Streamlit Cloud** pour l'hébergement gratuit
- **La communauté Python et Data Science** pour les outils open source utilisés

---

*Dernière mise à jour : Mai 2026 — Projet en cours de développement actif*