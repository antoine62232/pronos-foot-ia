# ====================================================================
# FICHIER : src/predictions_phase_eliminatoire.py
# ROLE    : Simuler la phase à élimination directe (du 1/16 à la finale)
# ====================================================================

import pandas as pd
import numpy as np
import joblib

# On récupère notre dictionnaire de points FIFA pour le modèle
import sys
sys.path.append('src')
from backtest import POINTS_FIFA

print("=" * 60)
print("🏆 SIMULATION DE LA PHASE ÉLIMINATOIRE — CdM 2026")
print("=" * 60)

# ====================================================================
# 1. CHARGEMENT DES OUTILS ET DES QUALIFIÉS
# ====================================================================
print("\n[1/4] Chargement du modèle IA et des équipes...")

modele = joblib.load("models/modele_football.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# On charge les qualifiés (on suppose que le CSV contient Groupe, Position, Equipe)
df_qualifies = pd.read_csv("data/qualifies_1_16.csv")

# On range les équipes dans des dictionnaires pour les retrouver facilement
# Ex: premiers['A'] donnera le 1er du Groupe A
premiers = df_qualifies[df_qualifies['qualification'].str.startswith('1er')].set_index('groupe')['equipe'].to_dict()
deuxiemes = df_qualifies[df_qualifies['qualification'].str.startswith('2eme')].set_index('groupe')['equipe'].to_dict()

# Pour les 3èmes, on les met dans une liste qu'on va "vider" au fur et à mesure
troisiemes = df_qualifies[df_qualifies['mode'] == 'repechage']['equipe'].tolist()

print(f"      {len(df_qualifies)} équipes prêtes pour le tableau final.")


# ====================================================================
# 2. FONCTION DE PRÉDICTION SANS MATCH NUL
# ====================================================================
def simuler_match_couperet(equipe_A, equipe_B):
    """
    Simule un match à élimination directe. 
    Force un vainqueur en cas de prédiction de match nul par l'IA.
    """
    fifa_A = POINTS_FIFA.get(equipe_A, 1200)
    fifa_B = POINTS_FIFA.get(equipe_B, 1200)
    
    # On simule un contexte "neutre" standard pour les formes
    donnees_match = pd.DataFrame([[
        1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 
        fifa_A, fifa_B, 1
    ]], columns=[
        'forme_attaque_domicile', 'forme_attaque_exterieur',
        'forme_defense_domicile', 'forme_defense_exterieur',
        'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
        'forme_defense_domicile_10', 'forme_defense_exterieur_10',
        'points_fifa_domicile', 'points_fifa_exterieur', 'match_neutre'
    ])
    
    probas = modele.predict_proba(donnees_match)[0]
    dict_probas = dict(zip(encoder.classes_, probas))
    
    proba_A = dict_probas.get('1', 0)
    proba_B = dict_probas.get('2', 0)
    
    # Le Tie-Break : on compare juste la proba de victoire de A vs B
    if proba_A >= proba_B:
        return equipe_A, (proba_A / (proba_A + proba_B)) * 100
    else:
        return equipe_B, (proba_B / (proba_A + proba_B)) * 100


# ====================================================================
# 3. CONSTRUCTION DU TABLEAU DES 1/16 DE FINALE (Règles FIFA)
# ====================================================================
print("\n[2/4] Création du tableau des 1/16 de finale...")

# On crée la liste exacte des 16 matchs selon ton calendrier
# .pop(0) permet de prendre le premier 3ème de notre liste et de l'enlever de la liste
matchs_1_16 = [
    (deuxiemes['A'], deuxiemes['B']),      # Match 1
    (premiers['C'], deuxiemes['F']),       # Match 2
    (premiers['E'], troisiemes.pop(0)),    # Match 3
    (premiers['F'], deuxiemes['C']),       # Match 4
    (deuxiemes['E'], deuxiemes['I']),      # Match 5
    (premiers['I'], troisiemes.pop(0)),    # Match 6
    (premiers['A'], troisiemes.pop(0)),    # Match 7
    (premiers['L'], troisiemes.pop(0)),    # Match 8
    (premiers['G'], troisiemes.pop(0)),    # Match 9
    (premiers['D'], troisiemes.pop(0)),    # Match 10
    (premiers['H'], deuxiemes['J']),       # Match 11
    (deuxiemes['K'], deuxiemes['L']),      # Match 12
    (premiers['B'], troisiemes.pop(0)),    # Match 13
    (deuxiemes['D'], deuxiemes['G']),      # Match 14
    (premiers['J'], deuxiemes['H']),       # Match 15
    (premiers['K'], troisiemes.pop(0))     # Match 16
]


# ====================================================================
# 4. SIMULATION TOUR PAR TOUR
# ====================================================================
print("\n[3/4] Début des matchs à élimination directe...\n")

noms_tours = ["1/16 de finale", "1/8 de finale", "Quarts de finale", "Demi-finales", "FINALE"]
matchs_tour_actuel = matchs_1_16
champion_final = ""

for tour in noms_tours:
    print("=" * 50)
    print(f" ⚽ {tour.upper()}")
    print("=" * 50)
    
    vainqueurs = []
    
    # On fait jouer chaque match
    for eq1, eq2 in matchs_tour_actuel:
        vainqueur, confiance = simuler_match_couperet(eq1, eq2)
        vainqueurs.append(vainqueur)
        print(f"   {eq1:18} vs {eq2:18} ➔ {vainqueur:15} ({confiance:.1f}%)")
        
    print("")
    
    # Préparation du tour suivant : on associe les vainqueurs 2 par 2
    if len(vainqueurs) > 1:
        matchs_tour_actuel = []
        for i in range(0, len(vainqueurs), 2):
            matchs_tour_actuel.append((vainqueurs[i], vainqueurs[i+1]))
    else:
        # Si on n'a plus qu'un vainqueur, c'est le champion !
        champion_final = vainqueurs[0]

# ====================================================================
# 5. LE VERDICT
# ====================================================================
print("\n" + "=" * 60)
print("🏆 LE VERDICT DE L'IA POUR LA COUPE DU MONDE 2026")
print("=" * 60)
print(f"\n   ⭐ CHAMPION DU MONDE : {champion_final.upper()} ⭐\n")

df_champion = pd.DataFrame([{"Role": "Champion", "Equipe": champion_final}])
df_champion.to_csv("data/vainqueur_final.csv", index=False)
print("[4/4] Vainqueur sauvegardé dans data/vainqueur_final.csv")