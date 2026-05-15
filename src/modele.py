#Rôle : Entraîner l'IA sur les matchs passés et sauvegarder le modèle pour l'utiliser dans l'application.

import pandas as pd
import os
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Chargement des données d'entraînement
print("Chargement des données d'entraînement...")
df = pd.read_csv("data/matchs_entrainement.csv")
print(f" {df.shape[0]} matchs chargés avec {df.shape[1]} colonnes.")

# Sélection des features (variables d'entrée)
features = [
    'forme_attaque_domicile', # Buts marqués en moyenne (5 derniers matchs)
    'forme_attaque_exterieur', # Buts marqués en moyenne (5 derniers matchs)
    'forme_defense_domicile', # Buts encaissés en moyenne (5 derniers matchs)
    'forme_defense_exterieur', # Buts encaissés en moyenne (5 derniers matchs)
    'points_fifa_domicile',    # Points FIFA officiels de l'équipe à domicile
    'points_fifa_exterieur'    # Points FIFA officiels de l'équipe à l'extérieur
]

X = df[features]
y = df['resultat'] # Variable cible (1, N, 2)

print(f"\nVariable cible (resultats) :")
print(y.value_counts()) # Affiche combien de fois chaque résultat apparaît

# Séparation entrainement/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"\nSéparation des données :")
print(f"Entraînement : {X_train.shape[0]} matchs")
print(f"Test : {X_test.shape[0]} matchs")

# Mise à l'échelle des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Données normalisées avec StandardScaler.")

# Création et entraînement du modèle
print("\nEntraînement du modèle en cours...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("Entraînement terminé")

# Évaluation du modèle
y_predictions = model.predict(X_test)
precision = accuracy_score(y_test, y_predictions)

print(f"\nResultats du modèle :")
print(f" Précision globale : {precision * 100:.1f}%")
print(f" (Sur {X_test.shape[0]} matchs de test)")

print("\nRapport détaillé :")
print(classification_report(y_test, y_predictions))

# Sauvegarde du modèle
os.makedirs("modeles", exist_ok=True)

chemin_modele = "models/modele_football.pkl"
dump(model, chemin_modele)
dump(scaler, "models/scaler.pkl")

print(f"\nModèle sauvegardé avec succès dans : {chemin_modele}")
print(f" Scaler sauvegardé avec succès dans : models/scaler.pkl")