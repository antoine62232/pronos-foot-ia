# ====================================================================
# ROLE : Recuperer les resultats reels des matchs via l'API-Football
# ====================================================================

import requests
import streamlit as st
import datetime


def _parser_matchs(data):
    """Extrait uniquement les matchs TERMINES (statut FT) d'une reponse API."""
    matchs = []
    for m in data.get("response", []):
        if m["fixture"]["status"]["short"] != "FT":
            continue  # on ignore les matchs pas termines (NS = pas commence, live, etc.)
        matchs.append({
            "equipe_dom": m["teams"]["home"]["name"],
            "equipe_ext": m["teams"]["away"]["name"],
            "score_dom":  m["goals"]["home"],
            "score_ext":  m["goals"]["away"],
            "ligue":      m["league"]["name"],
        })
    return matchs


# @st.cache_data : Streamlit GARDE le resultat en memoire pendant 15 min (ttl=900s)
# et le PARTAGE entre tous les visiteurs. => meme avec 500 visiteurs, on n'appelle
# l'API qu'une fois toutes les 15 min. C'est ce qui protege ton quota de 100/jour.
@st.cache_data(ttl=900)
def recuperer_matchs_termines(date):
    """
    Recupere les matchs termines d'une date donnee (format 'AAAA-MM-JJ').
    Renvoie une liste de dictionnaires (vide en cas de probleme reseau).
    """
    cle = st.secrets["API_FOOTBALL_KEY"]   # lue depuis secrets.toml, jamais en dur
    url = "https://v3.football.api-sports.io/fixtures"

    try:
        reponse = requests.get(
            url,
            headers={"x-apisports-key": cle},
            params={"date": date},
            timeout=10,   # on n'attend pas plus de 10s (evite que l'app se fige)
        )
        data = reponse.json()
    except Exception:
        return []   # souci reseau -> liste vide, l'app continue sans planter

    return _parser_matchs(data)

# Dictionnaire de correspondance : TON nom d'equipe  ->  nom utilise par l'API.
# A COMPLETER le 11 juin avec les vrais ecarts (ex: "South Korea": "Korea Republic").
# Si une equipe n'est pas ici, on suppose que les deux noms sont identiques.
CORRESPONDANCE_NOMS = {
    # "South Korea": "Korea Republic",
    # "Czech Republic": "Czechia",
}


def _nom_api(equipe):
    """Renvoie le nom tel que l'API l'ecrit (via le dico), sinon le nom inchange."""
    return CORRESPONDANCE_NOMS.get(equipe, equipe)


def _issue(score_dom, score_ext):
    """Traduit un score en resultat : '1' (dom), 'N' (nul) ou '2' (ext)."""
    if score_dom > score_ext:
        return "1"
    if score_dom == score_ext:
        return "N"
    return "2"


def comparer_predictions(df_predictions):
    """
    Compare chaque prediction de poule au resultat reel (si le match est joue).
    Renvoie (lignes, nb_corrects, nb_joues).
    - lignes : liste de dicts {dom, ext, prono, statut, score, correct}
    - statut : "joue" ou "attente"
    """
    aujourd_hui = datetime.date.today().isoformat()

    # 1) On rassemble les resultats reels des dates deja passees (les futures = 0 appel)
    index_reel = {}
    for d in sorted(set(df_predictions["date"].astype(str))):
        if d > aujourd_hui:
            continue  # match a venir : rien a recuperer
        for m in recuperer_matchs_termines(d):
            cle = frozenset({m["equipe_dom"], m["equipe_ext"]})
            index_reel[cle] = {"home": m["equipe_dom"], "sd": m["score_dom"], "se": m["score_ext"]}

    # 2) On compare chaque prediction au resultat reel correspondant
    lignes, nb_corrects, nb_joues = [], 0, 0
    for _, p in df_predictions.iterrows():
        api_dom, api_ext = _nom_api(p["equipe_dom"]), _nom_api(p["equipe_ext"])
        cle = frozenset({api_dom, api_ext})

        # Match futur OU resultat pas encore trouve -> en attente
        if str(p["date"]) > aujourd_hui or cle not in index_reel:
            lignes.append({"dom": p["equipe_dom"], "ext": p["equipe_ext"],
                           "prono": p["pronostic"], "statut": "attente",
                           "score": None, "correct": None})
            continue

        # On oriente le score selon TON domicile/exterieur (l'API peut les inverser)
        r = index_reel[cle]
        sd, se = (r["sd"], r["se"]) if r["home"] == api_dom else (r["se"], r["sd"])
        correct = (_issue(sd, se) == p["pronostic"])

        nb_joues += 1
        nb_corrects += int(correct)
        lignes.append({"dom": p["equipe_dom"], "ext": p["equipe_ext"],
                       "prono": p["pronostic"], "statut": "joue",
                       "score": (sd, se), "correct": correct})

    return lignes, nb_corrects, nb_joues

# ====================================================================
# L'ONGLET VISIBLE
# ====================================================================

def _texte_prono(prono, dom, ext):
    """Traduit le pronostic en texte lisible."""
    if prono == "1":
        return f"Victoire {dom}"
    if prono == "2":
        return f"Victoire {ext}"
    return "Match nul"


def afficher_onglet_resultats(df_predictions):
    """Onglet : confronte les pronostics de l'IA aux vrais resultats."""
    st.subheader("Réalité VS IA")
    st.markdown(
        '<p style="color: #94A3B8;">Le duel : les pronostics de l\'IA face aux vrais résultats, '
        'mis à jour automatiquement pendant la compétition.</p>',
        unsafe_allow_html=True
    )

    lignes, nb_corrects, nb_joues = comparer_predictions(df_predictions)

    # --- Le score global de l'IA ---
    if nb_joues == 0:
        st.info(
            "⏳ Aucun match de Coupe du Monde n'a encore été joué. "
            "Le score de l'IA s'affichera ici **automatiquement** dès le coup d'envoi (11 juin 2026) !"
        )
    else:
        pct = round(nb_corrects / nb_joues * 100)
        st.markdown(
            f'<div style="text-align: center; padding: 24px; margin-bottom: 24px;'
            f' background: linear-gradient(135deg, #1E1B4B, #141B2D);'
            f' border: 1px solid #1E293B; border-radius: 16px;">'
            f'  <div style="color: #94A3B8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Score de l\'IA</div>'
            f'  <div style="color: #A78BFA; font-size: 48px; font-weight: 800; margin: 6px 0;">{pct}%</div>'
            f'  <div style="color: #F8FAFC; font-size: 15px;">{nb_corrects} bons pronostics sur {nb_joues} matchs joués</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    joues = [l for l in lignes if l["statut"] == "joue"]
    attente = [l for l in lignes if l["statut"] == "attente"]

    # --- Les matchs joues (prediction vs realite) ---
    if joues:
        st.markdown("### Matchs joués")
        for l in joues:
            sd, se = l["score"]
            ok = l["correct"]
            couleur = "#10B981" if ok else "#F87171"
            badge = (f'<span style="color:{couleur}; font-weight:bold;">✅ Réussi</span>' if ok
                     else f'<span style="color:{couleur}; font-weight:bold;">❌ Manqué</span>')
            st.markdown(
                f'<div style="background:#141B2D; border:1px solid #1E293B; border-left:4px solid {couleur};'
                f' padding:14px; border-radius:8px; margin-bottom:10px;'
                f' display:flex; justify-content:space-between; align-items:center;">'
                f'  <div>'
                f'    <div style="color:#F8FAFC; font-weight:600;">{l["dom"]} {sd} - {se} {l["ext"]}</div>'
                f'    <div style="color:#94A3B8; font-size:12px; margin-top:2px;">Pronostic IA : {_texte_prono(l["prono"], l["dom"], l["ext"])}</div>'
                f'  </div>'
                f'  <div style="font-size:13px;">{badge}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # --- Les matchs a venir (replies) ---
    if attente:
        with st.expander(f"⏳ Matchs à venir ({len(attente)})"):
            for l in attente:
                st.markdown(
                    f'<div style="padding:6px 0; border-bottom:1px solid #1E293B;">'
                    f'<span style="color:#CBD5E1;">{l["dom"]} vs {l["ext"]}</span>'
                    f'<span style="color:#64748B; font-size:12px;"> — Pronostic IA : {_texte_prono(l["prono"], l["dom"], l["ext"])}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )