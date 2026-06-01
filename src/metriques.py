"""Metriques d'evaluation pour des predictions de resultats 1/N/2."""

import numpy as np


def evaluer(y_true, y_proba, afficher=True):
    """Calcule accuracy, log loss, Brier et RPS pour des predictions probabilistes.

    y_true  : resultats reels encodes (0 = domicile, 1 = nul, 2 = exterieur).
    y_proba : tableau (N, 3) de probabilites, colonnes [domicile, nul, exterieur].
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)

    # Borne les probas pour eviter log(0), puis renormalise chaque ligne a 1.
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), y_true] = 1

    accuracy = (y_proba.argmax(axis=1) == y_true).mean()
    log_loss = -np.log(y_proba[np.arange(n), y_true]).mean()
    brier = ((y_proba - y_onehot) ** 2).sum(axis=1).mean()

    # RPS : ecart entre probas cumulees et resultat cumule (tient compte de l'ordre 1/N/2).
    cum_proba = np.cumsum(y_proba, axis=1)
    cum_onehot = np.cumsum(y_onehot, axis=1)
    rps = (((cum_proba - cum_onehot) ** 2)[:, :2].sum(axis=1) / 2).mean()

    if afficher:
        print(f"Matchs : {n}")
        print(f"  Accuracy : {accuracy:.3f}")
        print(f"  Log loss : {log_loss:.3f}")
        print(f"  Brier    : {brier:.3f}")
        print(f"  RPS      : {rps:.3f}")

    return {"n": n, "accuracy": accuracy, "log_loss": log_loss, "brier": brier, "rps": rps}