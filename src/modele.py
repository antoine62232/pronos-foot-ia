#Rôle : Entraîner l'IA sur les matchs passés et sauvegarder le modèle pour l'utiliser dans l'application.

import pandas as pd
import os
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données d'entraînement
print("Chargement des données d'entraînement...")
df = pd.read_csv("data/matchs_entrainement.csv")
print(f" {df.shape[0]} matchs chargés avec {df.shape[1]} colonnes.")

# Sélection des features (variables d'entrée)
features = [
    'forme_attaque_domicile', # Moyenne des buts marqués à domicile sur les 5 derniers matchs
    'forme_attaque_exterieur' # Moyenne des buts marqués à l'extérieur sur les 5 derniers matchs
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

print(f"\nModèle sauvegardé avec succès dans : {chemin_modele}")