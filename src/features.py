import pandas as pd

df = pd.read_csv("data/matchs_entrainement.csv")

df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Fonction pour déterminer le résultat
def determiner_resultat(ligne):
    if ligne['home_score'] > ligne['away_score']:
        return "1"
    elif ligne['home_score'] == ligne['away_score']:
        return "N"
    else:
        return "2"

df['resultat'] = df.apply(determiner_resultat, axis=1)

print("Calcul de la forme récente des équipes en cours...")

# - groupby('home_team') : Sépare le tableau équipe par équipe
# - ['home_score'] : On regarde uniquement les buts marqués
# - rolling(5).mean() : Fait la moyenne sur les 5 derniers matchs
# - shift(1) : Décale le résultat d'un cran pour éviter que l'IA ne regarde dans le futur
df['forme_attaque_domicile'] = df.groupby('home_team')['home_score'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))

# Même chose pour les équipes à l'extérieur
df['forme_attaque_exterieur'] = df.groupby('away_team')['away_score'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))

# Remplace les valeurs manquantes par 0 (pour les premiers matchs où il n'y a pas assez de données pour calculer la moyenne)
df = df.fillna(0)
# On sauvegarde ce tableau
df.to_csv("data/matchs_entrainement.csv", index=False)

print("Création des features terminée avec succès !")

print("\nExemple sur le tout dernier matchs :")
print(df[['date', 'home_team', 'forme_attaque_domicile', 'away_team', 'forme_attaque_exterieur']].tail(1))