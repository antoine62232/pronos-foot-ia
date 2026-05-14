import pandas as pd

df = pd.read_csv("data/matchs_entrainement.csv")

df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# Fonction pour déterminer le résultat
def determiner_resultat(ligne):
    if ligne['home_score'] > ligne['away_score']:
        return "1"
    elif ligne['home_score'] == ligne['away_score']:
        return "N"
    else:
        return "2"

# On applique cette fonction à chaque ligne
df['resultat'] = df.apply(determiner_resultat, axis=1)

# On sauvegarde ce tableau
df.to_csv("data/matchs_entrainement.csv", index=False)

print("Création de la colonne 'resultat' terminée avec succès !")

print("\nVérification des 5 premiers matchs :")
print(df[['home_team', 'away_team', 'home_score', 'away_score', 'resultat']].head())