import pandas as pd

chemin_fichier = "data/matchs_internationaux.csv"
df = pd.read_csv(chemin_fichier)

# On affiche la liste complète de toutes les colonnes disponibles
print("🔍 Voici toutes les colonnes de notre base de données :")
print(df.columns.tolist())

# On cherche les matchs qui n'ont pas encore été joués
# Si un match n'a pas de score pour l'équipe à domicile (home_score), c'est qu'il est dans le futur
# "isna()" vérifie si la case est vide (is Not a Number)
# "sum()" compte combien de cases sont vides
matchs_futurs = df['home_score'].isna().sum()

print(f"\n🔮 Nombre de matchs futurs (sans score) : {matchs_futurs}")

# On calcule combien de matchs ont bien un score (matchs totaux - matchs futurs)
matchs_passes = df.shape[0] - matchs_futurs
print(f"✅ Nombre de matchs joués (avec score) pour entraîner l'IA : {matchs_passes}")