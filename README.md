# Pronos Foot IA — Coupe du Monde 2026

Bienvenue sur le projet **Pronos Foot IA**, une application de pronostics sportifs basée sur l'Intelligence Artificielle, ciblant la **Coupe du Monde de Football 2026**.

## Objectif du Projet

Développer une application web (Streamlit) capable de prédire les résultats des matchs internationaux en utilisant des modèles de Machine Learning entraînés sur plus de 25 ans de données.

---

## Stack Technique

- **Langage :** Python 3.11
- **Analyse de données :** Pandas, NumPy
- **IA / Machine Learning :** Scikit-Learn
- **Interface :** Streamlit
- **Gestion de version :** Git & GitHub

---

## Structure du Projet

pronos-foot-ia/
├── data/           # Bases de données (matchs passés et futurs)
├── models/         # Modèles d'IA entraînés (fichiers .pkl)
├── notebooks/      # Brouillons et analyses exploratoires
├── src/            # Scripts Python (collecte, nettoyage, features)
├── app.py          # Application Streamlit principale
├── requirements.txt # Dépendances du projet
└── .gitignore      # Fichiers exclus du suivi Git

Étapes Réalisées

Phase 1 : Environnement & Structure

- Configuration de l'environnement virtuel (venv).

- Initialisation de la structure de dossiers professionnelle.

- Mise en place du versioning Git avec branche dédiée preparation-donnees.

Phase 2 : Collecte de Données

- Création de src/collecte.py.

- Récupération de l'historique mondial des matchs internationaux (49 000+ lignes).

- Identification du calendrier officiel de la Coupe du Monde 2026 (72 matchs à prédire).

Phase 3 : Traitement & Feature Engineering

- Nettoyage : Filtrage pour ne garder que le football moderne (depuis l'an 2000) soit ~25 000 matchs.

- Variable Cible : Création de la colonne resultat (1, N, 2) basée sur les scores.

- Indicateurs IA (Features) : - Calcul de la moyenne de buts marqués sur les 5 derniers matchs (forme offensive).

Utilisation de .shift(1) pour éviter le Data Leakage (triche temporelle).

Phase 4 : Modélisation (En cours)

- Installation de scikit-learn.

- Création du premier modèle de base via src/modele.py (Régression Logistique).

- Mise en place de la validation par découpage Train/Test (80%/20%).

Comment lancer le projet ?

1. Activer l'environnement :

.\venv\Scripts\activate

2. Installer les outils :

pip install -r requirements.txt

3. Lancer la collecte et le traitement :

python src/collecte.py
python src/nettoyage.py
python src/features.py

4. Entraîner l'IA :

python src/modele.py

Journal de Bord

- Mai 2026 : Initialisation du projet, collecte des données internationales et création des premières features de forme récente. Utilisation des Conventional Commits (feat:).