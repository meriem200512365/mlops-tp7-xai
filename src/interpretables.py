import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xai_commun import CAS, X, Xte, Xtr, yte, ytr

res = {}


def note(nom, modele, Xt):
    """Évalue un modèle, stocke le score F1 et le nombre de fraudes détectées."""
    pred = modele.predict(Xt)
    res[nom] = {
        'f1': round(f1_score(yte, pred), 4),
        'fraudes_detectees': int(((pred == 1) & (yte == 1)).sum()),
    }
    print(
        f'{nom:28s} F1 = {res[nom]["f1"]:.4f} | '
        f'fraudes detectees : {res[nom]["fraudes_detectees"]} / {int(yte.sum())}'
    )


def logm(d):
    """Transforme le montant en échelle log10 (le montant agit par ordre de grandeur)."""
    d = d.copy()
    d['log_montant'] = np.log10(d.pop('montant'))
    return d


# 1. Évaluation du modèle de production (forêt aléatoire)
note('foret fraude@production', m, Xte)

# 2. Régression logistique avec montant brut
lr_brut = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
).fit(Xtr, ytr)
note('logistique, montant brut', lr_brut, Xte)

# 3. Régression logistique avec log10(montant)
lr = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
).fit(logm(Xtr), ytr)
note('logistique, log10(montant)', lr, logm(Xte))

# Analyse des coefficients standardisés
coef = {c: round(float(v), 2) for c, v in zip(logm(Xtr).columns, lr[-1].coef_[0])}
print('coefficients standardises :', coef)

# Calcul des multiplicateurs de cote (odds ratios)
cote = np.exp(lr[-1].coef_[0] / lr[0].scale_)
print(
    f"cote x {round(cote[0], 3)} par mois d'anciennete ; "
    f'x {round(cote[1], 2)} par transaction en 24 h'
)

# 4. Évaluation d'arbres de décision de différentes profondeurs
for p in [2, 3, 6, None]:
    t = DecisionTreeClassifier(max_depth=p, random_state=0).fit(Xtr, ytr)
    note(f'arbre profondeur {p} ({t.get_n_leaves()} feuilles)', t, Xte)

# 5. Analyse détaillée de l'arbre de profondeur 3
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
print(export_text(t3, feature_names=list(X.columns), decimals=1))
print(
    f'Mme Benali : feuille/nœud {t3.apply(CAS)[0]} | '
    f'P = {round(t3.predict_proba(CAS)[0, 1], 3)}'
)

# Sauvegarde des résultats au format JSON
with open('reports/interpretables.json', 'w') as f:
    json.dump(res, f, indent=2)