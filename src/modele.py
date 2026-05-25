import pandas as pd
import os
from joblib import dump
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight # Pour le rééquilibrage

# 1. Chargement des données
print("Chargement des données d'entraînement...")
df = pd.read_csv("data/matchs_entrainement.csv")
print(f" {df.shape[0]} matchs chargés.")

# 2. Les 11 features de notre V2 Officielle
features = [
    'forme_attaque_domicile', 'forme_attaque_exterieur',
    'forme_defense_domicile', 'forme_defense_exterieur',
    'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
    'forme_defense_domicile_10', 'forme_defense_exterieur_10',
    'points_fifa_domicile', 'points_fifa_exterieur',
    'match_neutre'
]

X = df[features]
y = df['resultat']

# 3. Encodage et rééquilibrage (très important pour éviter trop de nuls/victoires dom)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
poids = compute_sample_weight(class_weight='balanced', y=y_encoded)

X_train, X_test, y_train, y_test, poids_train, poids_test = train_test_split(
    X, y_encoded, poids, test_size=0.2, random_state=42, shuffle=False
)

# 4. Paramètres optimisés de notre modèle V2
print("\nEntraînement du modèle XGBoost V2...")
modele = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

modele.fit(X_train, y_train, sample_weight=poids_train)

# 5. Évaluation simple
y_pred = modele.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"\n Précision sur le jeu de test (V2) : {precision * 100:.1f}%")

# 6. Sauvegarde en dur du modèle
os.makedirs("models", exist_ok=True)
dump(modele, "models/modele_football.pkl")
dump(encoder, "models/label_encoder.pkl")
print("✅ Modèle V2 sauvegardé avec succès dans models/ !")