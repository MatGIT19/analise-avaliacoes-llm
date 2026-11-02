"""Dashboard das avaliações. Rode com: streamlit run dashboard.py"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.dados import baixar, carregar

st.set_page_config(page_title="Avaliações de clientes", page_icon="📊", layout="wide")

FAIXAS_ATRASO = ["8+ dias adiantado", "1 a 7 dias adiantado", "No dia ou até 3 dias de atraso",
                 "4 a 10 dias de atraso", "Mais de 10 dias de atraso"]
AZUL = "#2f6aa3"


def barras_horizontais(tabela: pd.DataFrame, categoria: str, valor: str, titulo_valor: str,
                       ordem: list[str] | None = None, formato: str = ".1f"):
    """Barras horizontais na ordem dada (ou do maior para o menor), com o valor na ponta."""
    eixo_y = alt.Y(f"{categoria}:N", sort=ordem or "-x", title=None, axis=alt.Axis(labelLimit=260))
    base = alt.Chart(tabela).encode(y=eixo_y, x=alt.X(f"{valor}:Q", title=titulo_valor))
    grafico = base.mark_bar(color=AZUL) + base.mark_text(align="left", dx=4).encode(
        text=alt.Text(f"{valor}:Q", format=formato)
    )
    st.altair_chart(grafico.properties(height=36 * len(tabela) + 40), width="stretch")
NOMES_MOTIVOS = {
    "atraso_na_entrega": "Atraso na entrega",
    "nao_recebido": "Não recebido",
    "produto_errado_ou_incompleto": "Produto errado ou incompleto",
    "produto_com_defeito": "Produto com defeito",
    "qualidade_abaixo_do_esperado": "Qualidade abaixo do esperado",
    "atendimento_do_vendedor": "Atendimento do vendedor",
    "outro": "Outro",
    "sem_reclamacao": "Sem reclamação",
    "invalido": "Resposta inválida",
}


@st.cache_data(show_spinner="Carregando avaliações...")
def dados() -> pd.DataFrame:
    baixar()  # no deploy, baixa os arquivos na primeira execução
    df = carregar()
    df["faixa_atraso"] = pd.cut(
        df["dias_de_atraso"], bins=[-1000, -8, -1, 3, 10, 1000], labels=FAIXAS_ATRASO
    )
    return df


df = dados()
st.title("📊 O que deixa o cliente insatisfeito?")
st.caption(
    "Avaliações com comentário do dataset público da Olist (e-commerce brasileiro, 2016–2018). "
    "Nota 1 ou 2 = avaliação negativa."
)

estados = st.multiselect("Filtrar por estado", sorted(df["estado"].dropna().unique()))
filtrado = df[df["estado"].isin(estados)] if estados else df
negativa = filtrado["nota"] <= 2

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avaliações com comentário", f"{len(filtrado):,}".replace(",", "."))
c2.metric("Negativas (nota 1–2)", f"{negativa.mean():.1%}")
c3.metric("Nota média: entregue no prazo", f"{filtrado.loc[~filtrado['atrasou'], 'nota'].mean():.2f}")
c4.metric("Nota média: entregue com atraso", f"{filtrado.loc[filtrado['atrasou'], 'nota'].mean():.2f}")

st.subheader("O atraso derruba a nota")
por_faixa = filtrado.groupby("faixa_atraso", observed=False)["nota"].agg(["mean", "size"])
por_faixa = por_faixa.reindex(FAIXAS_ATRASO).rename(columns={"mean": "nota", "size": "avaliacoes"})
barras_horizontais(por_faixa.rename_axis("faixa").reset_index(), "faixa", "nota",
                   "Nota média (1 a 5), pela data de entrega em relação ao prazo", ordem=FAIXAS_ATRASO, formato=".2f")
st.caption("Avaliações por faixa: " + " · ".join(f"{i}: {int(n):,}".replace(",", ".") for i, n in por_faixa["avaliacoes"].items()))

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Avaliações negativas por estado")
    por_estado = (
        df.assign(negativa=df["nota"] <= 2)
        .groupby("estado")
        .agg(negativas=("negativa", "mean"), avaliacoes=("negativa", "size"))
        .query("avaliacoes >= 200")
        .assign(negativas=lambda t: t["negativas"] * 100)
        .reset_index()
    )
    barras_horizontais(por_estado, "estado", "negativas", "% de avaliações negativas", formato=".0f")
    st.caption("Só estados com 200 avaliações ou mais; o filtro de estado não se aplica aqui.")
with col_b:
    st.subheader("Avaliações negativas por mês")
    por_mes = (
        filtrado.assign(negativa=negativa).groupby("mes").agg(negativas=("negativa", "mean"), n=("negativa", "size"))
    )
    por_mes = por_mes[por_mes["n"] >= 100]
    st.line_chart(por_mes["negativas"] * 100, x_label="", y_label="% negativas")
    st.caption("Só meses com 100 avaliações ou mais.")

st.subheader("Motivos das reclamações (classificados por LLM)")
arquivo_motivos = config.RESULTADOS_DIR / "motivos.csv"
if arquivo_motivos.exists():
    motivos = pd.read_csv(arquivo_motivos)
    if estados:
        motivos = motivos[motivos["estado"].isin(estados)]
    contagem = (motivos["motivo"].map(NOMES_MOTIVOS).value_counts(normalize=True) * 100).rename("pct")
    barras_horizontais(contagem.rename_axis("motivo").reset_index(), "motivo", "pct",
                       "% das avaliações negativas", formato=".0f")
    st.caption(f"Amostra de {len(motivos)} avaliações negativas classificadas pelo LLM.")
    with st.expander("Ver exemplos por motivo"):
        escolha = st.selectbox("Motivo", sorted(motivos["motivo"].unique()), format_func=NOMES_MOTIVOS.get)
        st.dataframe(motivos.loc[motivos["motivo"] == escolha, ["nota", "estado", "comentario"]].head(15),
                     hide_index=True, width="stretch")
else:
    st.info("Rode `python scripts/classificar_motivos.py` para gerar a classificação de motivos.")

st.subheader("Modelo clássico vs LLM")
arquivo_metricas = config.RESULTADOS_DIR / "metricas.json"
if arquivo_metricas.exists():
    m = json.loads(arquivo_metricas.read_text(encoding="utf-8"))
    linhas = [{"Modelo": "TF-IDF + regressão logística", **m["classico"]["amostra"],
               "ms por avaliação": m["classico"]["ms_por_avaliacao"]}]
    if "llm" in m:
        linhas.append({"Modelo": f"LLM ({m['llm']['modelo']})", **m["llm"]["amostra"],
                       "ms por avaliação": m["llm"]["ms_por_avaliacao"]})
    tabela = pd.DataFrame(linhas)[["Modelo", "acuracia", "f1_macro", "recall_negativa", "precisao_negativa", "ms por avaliação"]]
    st.dataframe(tabela.rename(columns={"acuracia": "Acurácia", "f1_macro": "F1 macro",
                                        "recall_negativa": "Recall negativas", "precisao_negativa": "Precisão negativas"}),
                 hide_index=True, width="stretch")
    st.caption(f"Mesma amostra balanceada de {m['dados']['amostra_comparacao']} avaliações do conjunto de teste.")
else:
    st.info("Rode `python scripts/comparar.py` para gerar a comparação.")
