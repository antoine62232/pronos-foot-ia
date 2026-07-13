# Prediktora — Prédictions Coupe du Monde 2026

Modèle de prédiction des matchs de la Coupe du Monde 2026, de la phase de groupes au vainqueur, à partir d'un classement Elo calculé sur l'historique international et de la forme récente des équipes.

[![Demo](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://prediktora.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Tester l'application](https://prediktora.streamlit.app/)**

---

## Performance

Le modèle est évalué par un backtest à séparation temporelle stricte : il apprend uniquement sur les matchs antérieurs à 2018, puis prédit 5 822 matchs de compétition joués ensuite (qualifications, Euro, Copa América, Ligue des Nations, CAN, Coupe du Monde). Aucune information postérieure n'entre dans l'entraînement.

L'évaluation ne se limite pas à l'accuracy. Elle inclut le Brier score, la log loss et le RPS, qui mesurent la qualité des probabilités et pas seulement le pronostic final.

| Type de matchs                     | Nombre | Accuracy   | Brier |
| ---------------------------------- | ------ | ---------- | ----- |
| Phases finales (matchs équilibrés) | 2 741  | **53,2 %** | 0,579 |
| Qualifications                     | 3 081  | 61,6 %     | 0,489 |
| Ensemble des compétitions          | 5 822  | 57,7 %     | 0,531 |

Repère : un pronostic au hasard donnerait environ 33 % d'accuracy et un Brier de 0,667.

Le chiffre représentatif pour la Coupe du Monde est celui des phases finales, 53,2 %. Ce sont des matchs équilibrés entre nations de niveau proche, sur terrain neutre, donc difficiles à prédire. Les qualifications, souvent déséquilibrées, gonflent mécaniquement la moyenne. C'est pour cette raison que les deux chiffres sont séparés.

**Calibration** : quand le modèle annonce 75 % de confiance, l'issue se réalise environ 76 % du temps ; à 89 %, environ 90 %. Il est légèrement prudent, ce qui est préférable à l'inverse.

---

## Fonctionnement

Classifieur XGBoost qui prédit l'issue d'un match en trois classes : victoire à domicile, match nul, victoire à l'extérieur.

Onze variables :

- **Forme récente** de chaque équipe : moyenne de buts marqués et encaissés sur les 5 et les 10 derniers matchs (forme à domicile pour l'équipe qui reçoit, à l'extérieur pour la visiteuse).
- **Classement Elo** calculé sur tout l'historique international. Le score de force de chaque équipe s'ajuste après chaque match selon l'écart entre le résultat attendu et le résultat obtenu : battre une équipe plus forte rapporte davantage. Le classement évolue dans le temps.
- **Contexte** : terrain neutre ou non.

L'Elo a remplacé les points FIFA, qui étaient figés dans le temps. Le gain est mesuré, et ce changement supprime un risque de fuite de données au backtest.

| Version              | Force d'équipe                | Accuracy (phases finales) |
| -------------------- | ----------------------------- | ------------------------- |
| Initiale             | Points FIFA (figés)           | 51,3 %                    |
| **Actuelle**         | **Elo (variable dans le temps)** | **53,2 %**             |

---

## Confrontation aux résultats réels

Pendant la compétition, chaque pronostic du modèle est comparé au vrai résultat.

Un workflow GitHub Actions s'exécute toutes les deux heures, interroge l'API football-data.org pour récupérer les scores des matchs terminés, et écrit dans `data/resultats_reels.csv`, committé dans le dépôt. L'application lit ce fichier plutôt que l'API.

Trois raisons à ce choix :

- **Quota d'API préservé** : quel que soit le nombre de visiteurs, ils lisent un fichier. Aucun appel API n'est déclenché côté application.
- **Indépendance vis-à-vis de la mise en veille** : les résultats restent à jour même quand l'application dort, puisque le workflow tourne chez GitHub.
- **Historique versionné** : chaque mise à jour est un commit, donc une trace datée.

L'onglet « Réalité VS IA » affiche le taux de réussite du modèle et le détail match par match.

---

## Fonctionnalités

- **Phase de groupes** : prédiction des 72 matchs, classement des 12 groupes, identification des 32 qualifiés (24 directs et 8 meilleurs troisièmes).
- **Phase à élimination directe** : simulation du bracket complet, avec départage automatique en cas de nul prédit.
- **Réalité VS IA** : suivi du taux de réussite pendant le tournoi.
- **Pronostics utilisateur** : possibilité de se comparer au modèle.
- **Probabilités affichées** pour chaque match.

---

## Stack

| Catégorie        | Outil             | Rôle                                |
| ---------------- | ----------------- | ----------------------------------- |
| Langage          | Python 3.11       | Backend                             |
| Manipulation     | Pandas            | DataFrames                          |
| Calculs          | NumPy             | Opérations vectorielles             |
| Machine Learning | XGBoost           | Classification 1/N/2                |
| Utilitaires ML   | Scikit-learn      | Encodage, pondération des classes   |
| Interface        | Streamlit         | Application web                     |
| Données live     | football-data.org | Scores du tournoi                   |
| Automatisation   | GitHub Actions    | Mise à jour planifiée des résultats |
| Hébergement      | Streamlit Cloud   | Mise en ligne                       |

---

## Architecture

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
│   └── resultats_reels.csv            # Resultats reels (rempli par le workflow)
│
├── src/
│   ├── elo.py                         # Classement Elo
│   ├── features.py                    # Variables de forme
│   ├── modele.py                      # Entrainement du modele de production
│   ├── metriques.py                   # Accuracy, log loss, Brier, RPS
│   ├── backtest_complet.py            # Backtest sur les matchs de competition
│   ├── comparer_features.py           # Comparaison de jeux de variables
│   ├── predictions_groupes.py         # Classements de groupes
│   └── predictions_phase_eliminatoire.py
│
├── components/                        # Interface Streamlit
├── scripts/
│   └── maj_resultats.py               # Recuperation des scores
├── .github/workflows/
│   └── maj-resultats.yml              # Planification
├── models/                            # Modele entraine (.pkl)
├── app.py
├── requirements.txt
└── LICENSE
```

---

## Installation

Prérequis : Python 3.11+, Git.

```bash
git clone https://github.com/antoine62232/pronos-foot-ia.git
cd pronos-foot-ia

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
streamlit run app.py
```

### Régénérer le modèle et les prédictions

```bash
python src/modele.py                            # Entrainement + Elo courant
python src/predictions_groupes.py               # Classements de groupes et qualifies
python src/predictions_phase_eliminatoire.py    # Bracket
python src/backtest_complet.py                  # Evaluation (optionnel)
```

---

## Suite

- Enrichir le modèle : cotes des bookmakers, valeur marchande des effectifs.
- Affiner l'Elo : différence de buts, importance du match.
- Travail UX/UI.

---

## À propos

Projet personnel mené en autonomie pour pratiquer le machine learning sur un sujet avec une échéance réelle. Premier contact avec la data science lors d'un stage de maintenance prédictive chez Stellantis (Python, scikit-learn, XGBoost).

Antoine Bayart 
[portfolio](https://portfolio-antoine-bayart.vercel.app)
[LinkedIn](https://www.linkedin.com/in/antoine-bayart/)

---

## Licence

MIT. Voir `LICENSE`.

## Sources

- football-data.co.uk et jeux de données publics de résultats internationaux : historique des matchs.
- football-data.org : scores en direct pendant la compétition.
