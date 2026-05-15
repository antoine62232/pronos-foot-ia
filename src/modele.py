import pandas as pd
import os
from joblib import dump
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données d'entraînement
print("Chargement des données d'entraînement...")
df = pd.read_csv("data/matchs_entrainement.csv")
print(f" {df.shape[0]} matchs chargés avec {df.shape[1]} colonnes.")

# Sélection des features (variables d'entrée)
features = [
    'forme_attaque_domicile',
    'forme_attaque_exterieur',
    'forme_defense_domicile',
    'forme_defense_exterieur',
    'points_fifa_domicile',
    'points_fifa_exterieur'
]

X = df[features]
y = df['resultat'] # Variable cible (1, N, 2)

print(f"\nVariable cible (resultats) :")
print(y.value_counts()) # Affiche combien de fois chaque résultat apparaît

# Traduction des étiquettes pour xgboost
encoder = LabelEncoder()
y = encoder.fit_transform(y) # Convertit les étiquettes en 0, 1, 2
# On garde en mémoire la correspondance pour pouvoir retraduire plus tard
# Ex : si XGBoost prédit "0", on saura que ça veut dire "1" (victoire domicile)
print(f"📖 Correspondance des étiquettes : {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

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

# CRÉATION DU MODÈLE XGBOOST AVEC DES PARAMÈTRES CHOISIS POUR ÉVITER LE SURAPPRENTISSAGE

# Paramètres choisis pour éviter le surapprentissage :
#
# n_estimators=200    → On construit 200 arbres successifs
# max_depth=5         → Chaque arbre fait maximum 5 niveaux de profondeur
#                       (plus profond = plus précis mais risque de surapprendre)
# learning_rate=0.1   → Vitesse d'apprentissage : 0.1 est un bon compromis
#                       (trop haut = chaotique, trop bas = lent)
# random_state=42     → Pour avoir toujours le même résultat
# eval_metric='mlogloss' → Métrique adaptée aux 3 classes (1/N/2)

modele = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

# Calcul des poids pour rééquilibrer les classes (car il y a plus de "1" que de "N" ou "2")
from sklearn.utils.class_weight import compute_sample_weight

poids = compute_sample_weight(class_weight='balanced', y=y_train)

modele.fit(X_train, y_train, sample_weight=poids)

print("Entraînement terminé")

# Évaluation du modèle
y_predictions = modele.predict(X_test)
# On retraduit les prédictions et les vraies valeurs en lettres
# pour que le rapport reste lisible (sinon on verrait 0/1/2 au lieu de 1/N/2)
y_predictions_lettres = encoder.inverse_transform(y_predictions)
y_test_lettres = encoder.inverse_transform(y_test)
precision = accuracy_score(y_test_lettres, y_predictions_lettres)

print(f"\nResultats du modèle :")
print(f" Précision globale : {precision * 100:.1f}%")
print(f" (Sur {X_test.shape[0]} matchs de test)")

print("\nRapport détaillé :")
print(classification_report(y_test_lettres, y_predictions_lettres))

# Importance des features
# On crée un tableau avec deux colonnes : le nom et son importance
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': modele.feature_importances_
})

# On trie de la plus importante à la moins importante
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print("\nImportance de chaque feature pour les prédictions :")
print(importances_df.to_string(index=False))

# Sauvegarde du modèle
os.makedirs("models", exist_ok=True)

chemin_modele = "models/modele_football.pkl"
dump(modele, chemin_modele)
dump(encoder, "models/label_encoder.pkl") # On sauvegarde aussi l'encodeur pour pouvoir retraduire les prédictions plus tard

print(f"\nModèle sauvegardé avec succès dans : {chemin_modele}")
print(f" Encoder sauvegardé avec succès dans : models/label_encoder.pkl")