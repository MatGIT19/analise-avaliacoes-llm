"""Modelo clássico: TF-IDF + regressão logística."""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def criar_modelo() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",  # "não" e "nao" viram a mesma palavra
                    ngram_range=(1, 2),  # bigramas pegam "não chegou", "não recomendo"
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def treinar(textos, rotulos) -> tuple[Pipeline, float]:
    """Devolve o modelo treinado e o tempo de treino em segundos."""
    modelo = criar_modelo()
    inicio = time.perf_counter()
    modelo.fit(textos, rotulos)
    return modelo, time.perf_counter() - inicio


def prever(modelo: Pipeline, textos) -> tuple[list[str], float]:
    """Devolve as previsões e o tempo médio por avaliação em milissegundos."""
    inicio = time.perf_counter()
    previsoes = list(modelo.predict(textos))
    ms = (time.perf_counter() - inicio) * 1000 / max(len(previsoes), 1)
    return previsoes, ms


def palavras_mais_fortes(modelo: Pipeline, n: int = 15) -> dict[str, list[str]]:
    """Termos que mais empurram para cada classe: ajuda a explicar o modelo."""
    vocab = modelo.named_steps["tfidf"].get_feature_names_out()
    logreg = modelo.named_steps["logreg"]
    pesos = logreg.coef_[0]  # positivo = empurra para classes_[1]
    ordem = pesos.argsort()
    return {
        str(logreg.classes_[0]): [vocab[i] for i in ordem[:n]],
        str(logreg.classes_[1]): [vocab[i] for i in ordem[::-1][:n]],
    }
