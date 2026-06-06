# ====================================================================
# ROLE : Petit utilitaire de dates (format francais), reutilisable partout.
# ====================================================================

import datetime

MOIS_FR = {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
           7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"}


def _date_fr(date_iso):
    """Convertit '2026-06-11' en '11 juin 2026'. Renvoie la chaine brute si souci."""
    try:
        d = datetime.date.fromisoformat(str(date_iso))
        return f"{d.day} {MOIS_FR[d.month]} {d.year}"
    except Exception:
        return str(date_iso)