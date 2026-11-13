
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from xai_commun import m, X, Xte, yte

# Calcul de l'importance par permutation
r = permutation_importance(
    m,
    Xte,
    yte,
    scoring='f1',
    n_repeats=10,
    random_state=0
)

# Comparaison entre MDI et permutation
t = pd.DataFrame({
    'MDI (foret)': m.feature_importances_,
    'permutation': r.importances_mean,
    'ecart-type': r.importances_std
}, index=X.columns).round(3)

# Trier les variables par importance décroissante
t = t.sort_values('permutation', ascending=False)

print(t.to_string())

# Sauvegarder les résultats
t.to_json('reports/importance.json')

# Créer le graphique
t['permutation'].plot.barh(
    xerr=t['ecart-type'],
    color='#4592EF'
)

plt.xlabel('perte de F1 quand la colonne est melangee')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('reports/permutation.png', dpi=120)
