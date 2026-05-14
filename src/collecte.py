import pandas as pd

url_internationale = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

print("Téléchargement de l'historique des matchs internationaux...")

# Pour lire le fichier sur internet
df = pd.read_csv(url_internationale)

# Sauvegarde des données dans le dossier data
df.to_csv("data/matchs_internationaux.csv", index=False)

print("Données sauvegardées avec succès dans le dossier data ")

# "df.shape" donne le nombre de lignes et de colonnes
print(f"\nNombre total de matchs téléchargés : {df.shape[0]}")

# On affiche les 5 dernières lignes (tail) pour voir les matchs les plus récents
print("\nAperçu des matchs les plus récents :")
print(df.tail())