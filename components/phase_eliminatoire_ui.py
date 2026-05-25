# ====================================================================
# ROLE    : Interface visuelle pour les classements et l'arbre final
# ====================================================================

import streamlit as st
import pandas as pd
from components.data_loader import get_url_drapeau, POINTS_FIFA

def _calculer_classements_groupes():
    """
    Lit les prédictions des 72 matchs et calcule le classement 
    exact (Points, Victoires, Nuls, Défaites) pour chaque groupe.
    """
    try:
        df_preds = pd.read_csv("data/predictions_groupes.csv")
        df_qualifies = pd.read_csv("data/qualifies_1_16.csv")
        liste_qualifies = df_qualifies['equipe'].tolist()
    except:
        return None, None

    # Initialisation des compteurs pour chaque équipe
    stats = {}
    equipe_to_groupe = {}

    for _, row in df_preds.iterrows():
        dom = row['home_team']
        ext = row['away_team']
        grp = row['group']
        prono = row['pronostic']

        for eq in [dom, ext]:
            if eq not in stats:
                stats[eq] = {'equipe': eq, 'J': 0, 'V': 0, 'N': 0, 'D': 0, 'Pts': 0, 'force': 0}
                equipe_to_groupe[eq] = grp

        stats[dom]['J'] += 1
        stats[ext]['J'] += 1
        
        # On utilise la probabilité brute comme "différence de buts" virtuelle en cas d'égalité
        stats[dom]['force'] += row['proba_1']
        stats[ext]['force'] += row['proba_2']

        # Attribution des points selon le pronostic de l'IA
        if prono == '1':
            stats[dom]['V'] += 1; stats[dom]['Pts'] += 3
            stats[ext]['D'] += 1
        elif prono == '2':
            stats[ext]['V'] += 1; stats[ext]['Pts'] += 3
            stats[dom]['D'] += 1
        else:
            stats[dom]['N'] += 1; stats[ext]['N'] += 1
            stats[dom]['Pts'] += 1; stats[ext]['Pts'] += 1

    # Regroupement par lettre (A à L)
    groupes = {lettre: [] for lettre in 'ABCDEFGHIJKL'}
    for eq, s in stats.items():
        groupes[equipe_to_groupe[eq]].append(s)

    # Tri de chaque groupe (par Points, puis par Force)
    for lettre in groupes:
        groupes[lettre] = sorted(groupes[lettre], key=lambda x: (x['Pts'], x['force']), reverse=True)

    return groupes, liste_qualifies


# --- FONCTION D'AFFICHAGE PRINCIPALE ---
def afficher_onglet_phase_eliminatoire(modele, encoder):
    """
    Affiche les classements de groupes sous forme de tableaux officiels
    puis le bracket des phases éliminatoires.
    """
    st.subheader("🏆 Tableau de Bord de la Phase Finale")
    st.markdown(
        '<p style="color: #94A3B8;">Découvrez les classements finaux des poules et le parcours prédictif de l\'IA.</p>',
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------------
    # SECTION 1 : AFFICHAGE DES 12 GROUPES (Design "Standing Table")
    # ----------------------------------------------------------------
    st.markdown("### 📊 Classements Finaux des Groupes")
    
    groupes_stats, liste_qualifies = _calculer_classements_groupes()
    
    if not groupes_stats:
        st.error("Veuillez d'abord exécuter le script des groupes dans votre terminal.")
        return

    groupes_liste = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    # Création des grilles (3 tableaux par ligne)
    for row_idx in range(0, 12, 3):
        cols = st.columns(3)
        for col_idx, groupe_lettre in enumerate(groupes_liste[row_idx:row_idx+3]):
            with cols[col_idx]:
                st.markdown(f"<h5 style='color: #A78BFA; margin-bottom: 5px;'>Groupe {groupe_lettre}</h5>", unsafe_allow_html=True)
                
                # Construction du design HTML sans espaces
                html = '<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px; text-align: center; background-color: #141B2D; border-radius: 8px; overflow: hidden; margin-bottom: 25px; border: 1px solid #1E293B;">'
                html += '<thead style="background-color: #0A0E1A; color: #94A3B8; font-size: 11px; text-transform: uppercase;">'
                html += '<tr><th style="padding: 8px 6px; text-align: left; width: 50%;">Équipe</th>'
                html += '<th style="padding: 8px 4px;">J</th><th style="padding: 8px 4px;">V</th><th style="padding: 8px 4px;">N</th><th style="padding: 8px 4px;">D</th>'
                html += '<th style="padding: 8px 6px; font-weight: bold; color: #F8FAFC;">Pts</th></tr></thead><tbody>'
                
                for idx, team_stat in enumerate(groupes_stats[groupe_lettre]):
                    team_name = team_stat['equipe']
                    is_q = team_name in liste_qualifies
                    
                    bg_color = "rgba(16, 185, 129, 0.08)" if is_q else "transparent"
                    border_left = "4px solid #10B981" if is_q else "4px solid transparent"
                    font_weight = "bold" if is_q else "normal"
                    color_text = "#F8FAFC" if is_q else "#CBD5E1"
                    
                    html += f'<tr style="border-top: 1px solid #1E293B; background-color: {bg_color};">'
                    html += f'<td style="padding: 8px 6px; text-align: left; border-left: {border_left};">'
                    html += f'<div style="display: flex; align-items: center; gap: 8px;">'
                    html += f'<span style="color: #64748B; font-size: 11px; width: 10px; text-align: right;">{idx+1}</span>'
                    html += f'<img src="{get_url_drapeau(team_name)}" width="20" style="border-radius: 2px; border: 0.5px solid #1E293B;">'
                    html += f'<span style="color: {color_text}; font-weight: {font_weight};">{team_name}</span>'
                    html += f'</div></td>'
                    html += f'<td style="padding: 8px 4px; color: #94A3B8;">{team_stat["J"]}</td>'
                    html += f'<td style="padding: 8px 4px; color: #94A3B8;">{team_stat["V"]}</td>'
                    html += f'<td style="padding: 8px 4px; color: #94A3B8;">{team_stat["N"]}</td>'
                    html += f'<td style="padding: 8px 4px; color: #94A3B8;">{team_stat["D"]}</td>'
                    html += f'<td style="padding: 8px 6px; font-weight: bold; color: #F8FAFC; font-size: 14px;">{team_stat["Pts"]}</td>'
                    html += f'</tr>'
                    
                html += "</tbody></table>"
                st.markdown(html, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0.5px solid #1E293B; margin-top: 10px; margin-bottom: 30px;'>", unsafe_allow_html=True)


    # ----------------------------------------------------------------
    # SECTION 2 : L'ARBRE DES PHASES ÉLIMINATOIRES (BRACKET)
    # ----------------------------------------------------------------
    st.markdown("### 🏆 L'Arbre des Matchs Couperets")
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "1/16 de Finale", "1/8 de Finale", "Quarts", "Demi-Finales", "🏆 LA FINALE"
    ])

    try:
        df_qualifies = pd.read_csv("data/qualifies_1_16.csv")
    except:
        return

    premiers = df_qualifies[df_qualifies['qualification'].str.startswith('1er')].set_index('groupe')['equipe'].to_dict()
    deuxiemes = df_qualifies[df_qualifies['qualification'].str.startswith('2eme')].set_index('groupe')['equipe'].to_dict()
    troisiemes = df_qualifies[df_qualifies['mode'] == 'repechage']['equipe'].tolist()

    matchs_1_16 = [
        (deuxiemes['A'], deuxiemes['B']), (premiers['C'], deuxiemes['F']),
        (premiers['E'], troisiemes[0]), (premiers['F'], deuxiemes['C']),
        (deuxiemes['E'], deuxiemes['I']), (premiers['I'], troisiemes[1]),
        (premiers['A'], troisiemes[2]), (premiers['L'], troisiemes[3]),
        (premiers['G'], troisiemes[4]), (premiers['D'], troisiemes[5]),
        (premiers['H'], deuxiemes['J']), (deuxiemes['K'], deuxiemes['L']),
        (premiers['B'], troisiemes[6]), (deuxiemes['D'], deuxiemes['G']),
        (premiers['J'], deuxiemes['H']), (premiers['K'], troisiemes[7])
    ]

    def _simuler_vainqueur(eq1, eq2):
        f1 = POINTS_FIFA.get(eq1, 1200)
        f2 = POINTS_FIFA.get(eq2, 1200)
        features_cols = [
            1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, f1, f2, 1
        ]
        df_m = pd.DataFrame([features_cols], columns=[
            'forme_attaque_domicile', 'forme_attaque_exterieur', 'forme_defense_domicile', 'forme_defense_exterieur',
            'forme_attaque_domicile_10', 'forme_attaque_exterieur_10', 'forme_defense_domicile_10', 'forme_defense_exterieur_10',
            'points_fifa_domicile', 'points_fifa_exterieur', 'match_neutre'
        ])
        p = modele.predict_proba(df_m)[0]
        if p[0] >= p[1]:
            return eq1, (p[0] / (p[0] + p[1])) * 100
        return eq2, (p[1] / (p[0] + p[1])) * 100

    v_1_16 = [_simuler_vainqueur(m[0], m[1]) for m in matchs_1_16]
    matchs_1_8 = [(v_1_16[i][0], v_1_16[i+1][0]) for i in range(0, 16, 2)]
    
    v_1_8 = [_simuler_vainqueur(m[0], m[1]) for m in matchs_1_8]
    matchs_1_4 = [(v_1_8[i][0], v_1_8[i+1][0]) for i in range(0, 8, 2)]
    
    v_1_4 = [_simuler_vainqueur(m[0], m[1]) for m in matchs_1_4]
    matchs_1_2 = [(v_1_4[i][0], v_1_4[i+1][0]) for i in range(0, 4, 2)]
    
    v_1_2 = [_simuler_vainqueur(m[0], m[1]) for m in matchs_1_2]
    match_finale = (v_1_2[0][0], v_1_2[1][0])
    
    champion, conf_champ = _simuler_vainqueur(match_finale[0], match_finale[1])

    def _creer_card_match(eq1, eq2, winner, conf):
        flag1, flag2 = get_url_drapeau(eq1), get_url_drapeau(eq2)
        w1_style = "color: #A78BFA; font-weight: bold;" if eq1 == winner else "color: #94A3B8;"
        w2_style = "color: #A78BFA; font-weight: bold;" if eq2 == winner else "color: #94A3B8;"
        
        st.markdown(
            f'<div style="background-color: #141B2D; border: 1px solid #1E293B; '
            f'padding: 12px; border-radius: 8px; margin-bottom: 10px;">'
            f'  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
            f'    <div style="display: flex; align-items: center; gap: 8px;">'
            f'       <img src="{flag1}" width="20" style="border-radius: 2px;">'
            f'       <span style="{w1_style}">{eq1}</span>'
            f'    </div>'
            f'    { "<span style=\'color: #10B981; font-size: 11px; font-weight: bold;\'>Qualifié</span>" if eq1 == winner else "" }'
            f'  </div>'
            f'  <div style="display: flex; justify-content: space-between; align-items: center;">'
            f'    <div style="display: flex; align-items: center; gap: 8px;">'
            f'       <img src="{flag2}" width="20" style="border-radius: 2px;">'
            f'       <span style="{w2_style}">{eq2}</span>'
            f'    </div>'
            f'    { "<span style=\'color: #10B981; font-size: 11px; font-weight: bold;\'>Qualifié</span>" if eq2 == winner else "" }'
            f'  </div>'
            f'  <div style="font-size: 11px; color: #64748B; margin-top: 8px; text-align: right; border-top: 1px dashed #1E293B; padding-top: 4px;">'
            f'     Confiance IA : {conf:.1f}%'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with sub_tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, m in enumerate(matchs_1_16):
            _creer_card_match(m[0], m[1], v_1_16[i][0], v_1_16[i][1])

    with sub_tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, m in enumerate(matchs_1_8):
            _creer_card_match(m[0], m[1], v_1_8[i][0], v_1_8[i][1])

    with sub_tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, m in enumerate(matchs_1_4):
            _creer_card_match(m[0], m[1], v_1_4[i][0], v_1_4[i][1])

    with sub_tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, m in enumerate(matchs_1_2):
            _creer_card_match(m[0], m[1], v_1_2[i][0], v_1_2[i][1])

    with sub_tab5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #FBBF24; margin-bottom: 20px;'>Grande Finale du 19 Juillet 2026</h4>", unsafe_allow_html=True)
        _creer_card_match(match_finale[0], match_finale[1], champion, conf_champ)
        
        flag_champ = get_url_drapeau(champion)
        st.markdown(
            f'<div style="text-align: center; margin-top: 30px; padding: 30px; '
            f'background: linear-gradient(135deg, #1E1B4B, #141B2D); border: 2px solid #FBBF24; border-radius: 16px; box-shadow: 0px 10px 30px rgba(251, 191, 36, 0.15);">'
            f'  <h2 style="color: #FBBF24; margin-top:0; letter-spacing: 2px;">🌟 CHAMPION DU MONDE 🌟</h2>'
            f'  <img src="{flag_champ}" width="140" style="border-radius: 8px; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); margin: 20px 0;">'
            f'  <h1 style="color: #F8FAFC; margin-bottom: 0; font-size: 42px;">{champion.upper()}</h1>'
            f'</div>',
            unsafe_allow_html=True
        )