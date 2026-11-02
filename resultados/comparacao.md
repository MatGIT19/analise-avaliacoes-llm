# Comparação: modelo clássico vs LLM

Amostra de **400 avaliações** do conjunto de teste, metade negativas e metade positivas. Rótulo real = nota do cliente (1-2 negativa, 4-5 positiva).

| Modelo | Acurácia | F1 macro | Recall negativas | Precisão negativas | Tempo por 1.000 | Custo por 1.000 |
|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 95.5% | 0.955 | 98.5% | 92.9% | 0.02 s | ~US$ 0 (roda na CPU) |

No conjunto de teste completo (7689 avaliações, proporção real de classes), o modelo clássico teve acurácia de 93.7% e F1 macro de 0.925.
