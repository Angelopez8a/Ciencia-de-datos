# Módulo 2 · Ensambles: votación, bagging, bosque aleatorio, boosting y stacking
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier)
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
arbol = DecisionTreeClassifier(max_depth=4, random_state=0)

modelos = {
    "Árbol individual":       arbol,
    "Votación dura":          VotingClassifier([("lr", logit), ("knn", knn), ("dt", arbol)], voting="hard"),
    "Votación suave":         VotingClassifier([("lr", logit), ("knn", knn), ("dt", arbol)], voting="soft"),
    "Bagging (100 árboles)":  BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0),
    "Bosque aleatorio":       RandomForestClassifier(n_estimators=200, random_state=0),
    "AdaBoost (tocones)":     AdaBoostClassifier(n_estimators=200, random_state=0),
    "Gradient Boosting":      GradientBoostingClassifier(random_state=0),
    "Stacking":               StackingClassifier([("knn", knn), ("dt", arbol), ("rf", RandomForestClassifier(100, random_state=0))],
                                                 final_estimator=LogisticRegression(max_iter=1000)),
}
print(f"{'Modelo':<24}{'Accuracy':>10}{'AUC':>8}")
for nombre, m in modelos.items():
    m.fit(Xt, yt)
    acc = m.score(Xv, yv)
    if getattr(m, "voting", "soft") == "hard":          # el voto duro no da probabilidades
        auc_txt = "n/a"
    else:
        auc_txt = f"{roc_auc_score(yv, m.predict_proba(Xv)[:, 1]):.3f}"
    print(f"{nombre:<24}{acc:>10.3f}{auc_txt:>8}")
