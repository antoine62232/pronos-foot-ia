# ====================================================================
# ROLE : Petit bandeau annoncant la nouveaute "avant-tournoi / live".
#        Masquable pour la session en cours (reapparait au rechargement).
# ====================================================================

import streamlit as st


def afficher_banniere_nouveaute():
    if st.session_state.get("nouveaute_live_fermee"):
        return

    st.markdown(
        '<div style="background-color:#141B2D; border:1px solid #1E293B; border-left:4px solid #A78BFA;'
        ' border-radius:10px; padding:14px 18px; margin-bottom:14px;">'
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

    # Bouton discret pour masquer le bandeau (le temps de la session)
    if st.button("Masquer", key="masquer_nouveaute"):
        st.session_state["nouveaute_live_fermee"] = True
        st.rerun()