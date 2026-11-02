from types import SimpleNamespace

import pytest

from conftest import ModelosFalsos
from src import llm as modulo
from src.llm import ClassificadorLLM


@pytest.fixture(autouse=True)
def sem_espera(monkeypatch):
    monkeypatch.setattr(modulo.time, "sleep", lambda s: None)


def test_classifica_em_lotes_e_mantem_a_ordem(cliente_falso):
    c = ClassificadorLLM(cliente_falso, lote=2, pausa_s=0)
    r = c.classificar(["a", "b", "c"], ["Amei, recomendo", "Não recebi", "Péssimo produto"])
    assert [x["review_id"] for x in r] == ["a", "b", "c"]
    assert [x["sentimento"] for x in r] == ["positiva", "negativa", "negativa"]
    assert r[1]["motivo"] == "nao_recebido"
    assert cliente_falso.models.chamadas == 2  # 3 avaliações em lotes de 2
    assert c.uso.tokens_entrada == 200 and c.uso.custo_estimado_usd() > 0


def test_cache_evita_chamar_de_novo(cliente_falso, tmp_path):
    cache = tmp_path / "cache.jsonl"
    ClassificadorLLM(cliente_falso, pausa_s=0, cache=cache).classificar(["a"], ["Amei"])
    novo_cliente = SimpleNamespace(models=ModelosFalsos())
    r = ClassificadorLLM(novo_cliente, pausa_s=0, cache=cache).classificar(["a"], ["Amei"])
    assert r[0]["sentimento"] == "positiva"
    assert novo_cliente.models.chamadas == 0


def test_tenta_de_novo_quando_a_api_limita():
    cliente = SimpleNamespace(models=ModelosFalsos(falhas_antes=2))
    r = ClassificadorLLM(cliente, pausa_s=0).classificar(["a"], ["Amei"])
    assert r[0]["sentimento"] == "positiva"
    assert cliente.models.chamadas == 3


def test_json_invalido_vira_invalido_sem_quebrar():
    cliente = SimpleNamespace(models=ModelosFalsos(resposta_quebrada=True))
    c = ClassificadorLLM(cliente, pausa_s=0)
    r = c.classificar(["a"], ["Amei"])
    assert r[0]["sentimento"] == "invalido"
    assert c.uso.erros_de_formato == 1
