from src.classico import palavras_mais_fortes, prever, treinar
from src.dados import amostra_balanceada, carregar, rotular_sentimento
from src.metricas import resumo


def test_carregar_junta_tabelas_e_descarta_sem_comentario(pasta_olist):
    df = carregar(pasta_olist)
    assert len(df) == 61  # 60 + a nota 3; as 4 sem comentário saem
    assert set(df["estado"]) == {"SP", "RJ"}
    assert df.loc[df["review_id"] == "r0", "comentario"].item() == "Título. Chegou rápido, recomendo"


def test_atraso_calculado_pela_data_prevista(pasta_olist):
    df = carregar(pasta_olist).set_index("review_id")
    assert df.loc["r0", "dias_de_atraso"] < 0 and not df.loc["r0", "atrasou"]
    assert df.loc["r1", "dias_de_atraso"] > 0 and df.loc["r1", "atrasou"]


def test_rotulo_tira_nota_3(pasta_olist):
    rotulado = rotular_sentimento(carregar(pasta_olist))
    assert "neutra" not in set(rotulado["review_id"])
    assert rotulado.loc[rotulado["nota"] == 1, "sentimento"].eq("negativa").all()


def test_amostra_balanceada_tem_metade_de_cada(pasta_olist):
    amostra = amostra_balanceada(rotular_sentimento(carregar(pasta_olist)), 20)
    assert amostra["sentimento"].value_counts().to_dict() == {"negativa": 10, "positiva": 10}


def test_modelo_classico_aprende_o_basico(pasta_olist):
    dados = rotular_sentimento(carregar(pasta_olist))
    modelo, _ = treinar(dados["comentario"], dados["sentimento"])
    previsoes, ms = prever(modelo, ["péssimo, não recomendo", "excelente, amei"])
    assert previsoes == ["negativa", "positiva"]
    assert ms >= 0
    assert set(palavras_mais_fortes(modelo, 3)) == {"negativa", "positiva"}


def test_metricas_contam_respostas_invalidas():
    m = resumo(["negativa", "positiva", "negativa"], ["negativa", "positiva", "invalido"])
    assert m["respostas_invalidas"] == 1
    assert m["acuracia"] == round(2 / 3, 4)
    assert m["matriz_confusao"]["valores"] == [[1, 0], [0, 1]]
