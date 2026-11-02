"""Dados sintéticos no formato da Olist e um cliente falso do Gemini."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

POSITIVOS = ["Chegou rápido, recomendo", "Excelente produto, amei", "Ótimo, chegou antes do prazo",
             "Produto lindo e bem embalado", "Gostei muito, recomendo a loja"]
NEGATIVOS = ["Não recebi o produto", "Péssimo, veio quebrado", "Não recomendo, veio errado",
             "Produto ruim e atrasou muito", "Até hoje não chegou, péssimo"]


@pytest.fixture
def pasta_olist(tmp_path):
    """60 avaliações (metade com nota 5, metade com nota 1) + 4 sem comentário + 1 nota 3."""
    avaliacoes, pedidos, clientes = [], [], []
    for i in range(60):
        positiva = i % 2 == 0
        texto = (POSITIVOS if positiva else NEGATIVOS)[i % 5]
        avaliacoes.append({
            "review_id": f"r{i}", "order_id": f"o{i}", "review_score": 5 if positiva else 1,
            "review_comment_title": "Título" if i == 0 else None, "review_comment_message": texto,
            "review_creation_date": f"2018-0{1 + i % 3}-10 00:00:00", "review_answer_timestamp": "2018-01-11 00:00:00",
        })
        entrega = "2018-01-05 10:00:00" if positiva else "2018-01-25 10:00:00"
        pedidos.append({
            "order_id": f"o{i}", "customer_id": f"c{i}", "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-01 10:00:00", "order_approved_at": None,
            "order_delivered_carrier_date": None, "order_delivered_customer_date": entrega,
            "order_estimated_delivery_date": "2018-01-15 00:00:00",
        })
        clientes.append({"customer_id": f"c{i}", "customer_unique_id": f"u{i}", "customer_zip_code_prefix": 1000,
                         "customer_city": "sao paulo", "customer_state": "SP" if i < 30 else "RJ"})
    for j in range(4):  # sem comentário: devem sair
        avaliacoes.append({"review_id": f"vazia{j}", "order_id": "o0", "review_score": 5,
                           "review_comment_title": None, "review_comment_message": None,
                           "review_creation_date": "2018-01-10 00:00:00", "review_answer_timestamp": None})
    avaliacoes.append({"review_id": "neutra", "order_id": "o1", "review_score": 3, "review_comment_title": None,
                       "review_comment_message": "Mais ou menos", "review_creation_date": "2018-01-10 00:00:00",
                       "review_answer_timestamp": None})
    pd.DataFrame(avaliacoes).to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame(pedidos).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(clientes).to_csv(tmp_path / "olist_customers_dataset.csv", index=False)
    return tmp_path


class ModelosFalsos:
    """Classifica por palavra-chave e devolve JSON como o Gemini; pode falhar antes de responder."""

    def __init__(self, falhas_antes=0, resposta_quebrada=False):
        self.chamadas = 0
        self.falhas_antes = falhas_antes
        self.resposta_quebrada = resposta_quebrada

    def generate_content(self, model, contents, config):
        self.chamadas += 1
        if self.falhas_antes:
            self.falhas_antes -= 1
            erro = RuntimeError("limite")
            erro.code = 429
            raise erro
        if self.resposta_quebrada:
            return SimpleNamespace(text="isto não é json", usage_metadata=None)
        itens = []
        for linha in contents.splitlines():
            numero, texto = linha.split(". ", 1)
            negativa = any(p in texto.lower() for p in ("não", "péssimo", "ruim", "quebrado"))
            itens.append({"id": int(numero), "sentimento": "negativa" if negativa else "positiva",
                          "motivo": "nao_recebido" if "receb" in texto.lower() or "chegou" in texto.lower() and negativa
                          else ("outro" if negativa else "sem_reclamacao")})
        uso = SimpleNamespace(prompt_token_count=100, candidates_token_count=20)
        return SimpleNamespace(text=json.dumps(itens), usage_metadata=uso)


@pytest.fixture
def cliente_falso():
    return SimpleNamespace(models=ModelosFalsos())
