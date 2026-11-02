"""Métricas usadas para comparar os dois classificadores."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

CLASSES = ["negativa", "positiva"]


def resumo(verdadeiro, previsto) -> dict:
    verdadeiro, previsto = list(verdadeiro), list(previsto)
    validos = [(v, p) for v, p in zip(verdadeiro, previsto) if p in CLASSES]
    v, p = zip(*validos) if validos else ([], [])
    matriz = confusion_matrix(v, p, labels=CLASSES).tolist() if validos else [[0, 0], [0, 0]]
    return {
        "n": len(verdadeiro),
        "respostas_invalidas": len(verdadeiro) - len(validos),
        "acuracia": round(accuracy_score(verdadeiro, previsto), 4),
        "f1_macro": round(f1_score(v, p, labels=CLASSES, average="macro", zero_division=0), 4),
        "precisao_negativa": round(precision_score(v, p, pos_label="negativa", zero_division=0), 4),
        "recall_negativa": round(recall_score(v, p, pos_label="negativa", zero_division=0), 4),
        "matriz_confusao": {"linhas=real, colunas=previsto": CLASSES, "valores": matriz},
    }
