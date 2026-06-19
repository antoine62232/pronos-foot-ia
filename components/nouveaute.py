# ====================================================================
# ROLE : Petit bandeau annoncant la nouveaute "avant-tournoi / live".
#        Masquable pour la session en cours (reapparait au rechargement)
#        via une petite croix dans le coin du bandeau.
# ====================================================================

import streamlit as st


def afficher_banniere_nouveaute():
    if st.session_state.get("nouveaute_live_fermee"):
        return

    # La croix est placee en absolu dans le coin du bandeau : pour ca, le conteneur
    # qui les regroupe est en position:relative (= repere pour le positionnement).
    st.markdown("""
    <style>
      .st-key-banniere_nouveaute { position: relative; }
      .st-key-fermer_nouveaute {
          position: absolute; top: 8px; right: 8px; z-index: 5; width: auto !important;
      }
      .st-key-fermer_nouveaute button {
          background: transparent !important; border: none !important; box-shadow: none !important;
          color: #64748B !important; padding: 0 6px !important;
          min-height: 0 !important; height: auto !important;
          font-size: 16px !important; line-height: 1 !important;
      }
      .st-key-fermer_nouveaute button:hover { color: #F8FAFC !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="banniere_nouveaute"):
        st.markdown(
            '<div style="background-color:#141B2D; border:1px solid #1E293B; border-left:4px solid #A78BFA;'
            ' border-radius:10px; padding:14px 44px 14px 18px; margin-bottom:14px;">'
            '  <p style="color:#CBD5E1; font-size:14px; line-height:1.6; margin:0;">'
            '    <span style="background:#A78BFA; color:#0F172A; font-weight:700; font-size:11px;'
            '          padding:2px 8px; border-radius:6px; margin-right:8px;">NOUVEAU</span>'
            '    Tu peux maintenant comparer les prédictions <strong style="color:#F8FAFC;">d\'avant-tournoi</strong>'
            '    avec la version <strong style="color:#F8FAFC;">mise à jour en direct</strong>, recalculée avec les'
            '    vrais résultats déjà joués. Sur les onglets <em>Phase de groupes</em> et <em>Phase Éliminatoire</em>,'
            '    un sélecteur « Avant le tournoi / Mise à jour live » te laisse basculer d\'un clic.'
            '  </p>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("✕", key="fermer_nouveaute", help="Masquer"):
            st.session_state["nouveaute_live_fermee"] = True
            st.rerun()