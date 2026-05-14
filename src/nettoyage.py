import pandas as pd

# Chargement de la BDD brutes
df = pd.read_csv("data/matchs_internationaux.csv")

df['date'] = pd.to_datetime(df['date'])

# Filtrage de la BDD pour ne garder que les matchs à partir de l'an 2000
# "dt.year" permet d'extraire juste l'année de notre colonne date
df_moderne = df[df['date'].dt.year >= 2000]

# Séparation des matchs du passé et des matchs du futur
# "~" (tilde) = "le contraire de". 
# Donc ici : on garde les lignes où le score à domicile N'EST PAS vide
matchs_passes = df_moderne[~df_moderne['home_score'].isna()]

# Ici : on garde les lignes où le score à domicile EST vide
matchs_futurs = df_moderne[df_moderne['home_score'].isna()]

# Sauvegarde des deux nouveaux tableaux dans le dossier data
matchs_passes.to_csv("data/matchs_entrainement.csv", index=False)
matchs_futurs.to_csv("data/matchs_a_predire.csv", index=False)

print(f" Nettoyage terminé !")
print(f" Matchs d'entraînement (depuis 2000) : {matchs_passes.shape[0]} matchs.")
print(f" Matchs de la Coupe du Monde à prédire : {matchs_futurs.shape[0]} matchs.")