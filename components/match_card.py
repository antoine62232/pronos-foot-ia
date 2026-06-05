import streamlit as st
from components.data_loader import get_url_drapeau, get_rang_fifa

# FONCTION PRINCIPALE

def afficher_carte_match(match):
    """
    Affiche une carte stylée pour un match donné.

    Paramètre :
        match : Une ligne du DataFrame df_predictions
    """

    # Préparation des données (extraction et formatage)
    eq_dom = match['equipe_dom']
    eq_ext = match['equipe_ext']
    url_drap_dom = get_url_drapeau(eq_dom)
    url_drap_ext = get_url_drapeau(eq_ext)
    rang_dom = get_rang_fifa(eq_dom)
    rang_ext = get_rang_fifa(eq_ext)
    p1 = match['proba_1']
    pN = match['proba_N']
    p2 = match['proba_2']

    pronostic_texte = formater_pronostic(match['pronostic'], eq_dom, eq_ext)
    rang_dom_txt = _rang_texte(rang_dom)
    rang_ext_txt = _rang_texte(rang_ext)
    couleur_1, couleur_N, couleur_2 = _calculer_couleurs_barres(match['pronostic'])

    # Construction du HTML (sur une seule ligne logique)
    # IMPORTANT : on construit le HTML morceau par morceau pour éviter les problèmes d'indentation qui font foirer st.markdown

    html = (
        f'<div style="background: #141B2D; padding: 20px; border-radius: 12px;'
        f' border: 0.5px solid #1E293B; margin-bottom: 16px;'
        f' position: relative; overflow: hidden;">'
        f'<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px;'
        f' background: linear-gradient(90deg, #A78BFA, #22D3EE);"></div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">'
        f'<span style="color: #94A3B8; font-size: 12px;">📅 {match["date"]}</span>'
        f'<span style="background: rgba(34,211,238,0.15); color: #22D3EE; padding: 3px 10px;'
        f' border-radius: 6px; font-size: 10px; font-weight: 500;'
        f' border: 0.5px solid rgba(34,211,238,0.3);">COUPE DU MONDE 2026</span>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">'
        f'<div style="display: flex; align-items: center; gap: 12px; flex: 1;">'
        f'<img src="{url_drap_dom}" style="width: 40px; height: auto; border-radius: 3px; border: 0.5px solid #1E293B;">'
        f'<div>'
        f'<div style="color: #F8FAFC; font-weight: 600; font-size: 16px;">{eq_dom}</div>'
        f'<div style="color: #94A3B8; font-size: 11px;">{rang_dom_txt}</div>'
        f'</div>'
        f'</div>'
        f'<div style="text-align: center; color: #475569; font-size: 12px; font-weight: 500;">VS</div>'
        f'<div style="display: flex; align-items: center; gap: 12px; flex: 1; justify-content: flex-end;">'
        f'<div style="text-align: right;">'
        f'<div style="color: #F8FAFC; font-weight: 600; font-size: 16px;">{eq_ext}</div>'
        f'<div style="color: #94A3B8; font-size: 11px;">{rang_ext_txt}</div>'
        f'</div>'
        f'<img src="{url_drap_ext}" style="width: 40px; height: auto; border-radius: 3px; border: 0.5px solid #1E293B;">'
        f'</div>'
        f'</div>'
        f'<div style="background: #0A0E1A; padding: 14px; border-radius: 8px; border: 0.5px solid #1E293B;">'
        f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">'
        f'<span style="color: #94A3B8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">🤖 L\'IA prédit</span>'
        f'<span style="background: linear-gradient(135deg, #A78BFA, #22D3EE); color: white;'
        f' padding: 3px 10px; border-radius: 6px; font-size: 11px;'
        f' font-weight: 500; margin-left: auto;">{pronostic_texte}</span>'
        f'</div>'
        f'<div style="display: flex; gap: 4px; height: 8px;">'
        f'<div style="flex: {p1}; background: {couleur_1}; border-radius: 4px;"></div>'
        f'<div style="flex: {pN}; background: {couleur_N}; border-radius: 4px;"></div>'
        f'<div style="flex: {p2}; background: {couleur_2}; border-radius: 4px;"></div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">'
        f'<span style="color: #94A3B8;">{eq_dom} <strong style="color: #F8FAFC;">{p1:.0f}%</strong></span>'
        f'<span style="color: #FBBF24;">Nul <strong>{pN:.0f}%</strong></span>'
        f'<span style="color: #A78BFA;">{eq_ext} <strong>{p2:.0f}%</strong></span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    # On affiche le HTML construit
    st.markdown(html, unsafe_allow_html=True)

# FONCTIONS INTERNES (helpers)

def formater_pronostic(pronostic, equipe_dom, equipe_ext):
    """Convertit le pronostic brut en texte lisible."""
    if pronostic == '1':
        return f"Victoire {equipe_dom}"
    elif pronostic == '2':
        return f"Victoire {equipe_ext}"
    else:
        return "Match nul"


def _calculer_couleurs_barres(pronostic):
    """Détermine les couleurs des 3 barres de probabilité."""
    GRADIENT_ACTIF = "linear-gradient(90deg, #A78BFA, #22D3EE)"
    JAUNE_ACTIF    = "#FBBF24"
    GRIS_ETEINT    = "#334155"

    if pronostic == '1':
        return GRADIENT_ACTIF, GRIS_ETEINT, GRIS_ETEINT
    elif pronostic == '2':
        return GRIS_ETEINT, GRIS_ETEINT, GRADIENT_ACTIF
    else:
        return GRIS_ETEINT, JAUNE_ACTIF, GRIS_ETEINT
    
def _rang_texte(rang):
    """Texte du rang mondial, avec '1er' au lieu de '1ème'."""
    if not rang:
        return "Non classé"
    return f"{rang}er mondial" if rang == 1 else f"{rang}ème mondial"