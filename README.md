# 📊 Analisador de Avaliações: LLM vs ML clássico

> 🚧 **Em desenvolvimento.** Este repositório tem a estrutura e o plano do projeto; o código e os resultados estão sendo adicionados. Acompanhe o progresso no [roadmap](#roadmap).

Classificação de avaliações reais de clientes de e-commerce brasileiro por **sentimento** e **motivo da reclamação**, comparando um modelo de machine learning tradicional com um LLM. O resultado vira um dashboard com os principais motivos de insatisfação por estado e por mês.

## Pergunta que o projeto responde

Para classificar milhares de avaliações de clientes, vale mais a pena um modelo clássico treinado nos dados (barato e rápido) ou um LLM com um bom prompt (sem treino e capaz de explicar o motivo)? Em quais casos cada um erra?

## Dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), no Kaggle: cerca de 99 mil pedidos e 99 mil avaliações com nota de 1 a 5 e comentário em português. Licença CC BY-NC-SA 4.0. Como baixar: [data/README.md](data/README.md).

## Abordagem planejada

| Etapa | O que será feito |
|---|---|
| Preparação | Juntar avaliações, pedidos e clientes; manter só avaliações com comentário; rótulo **negativa** (nota 1 ou 2) ou **positiva** (nota 4 ou 5) |
| Modelo clássico | TF-IDF + regressão logística, com divisão treino e teste |
| LLM | Prompt com exemplos (few-shot) e saída em JSON: sentimento e motivo (atraso, produto errado, defeito, atendimento, outro) |
| Comparação | Mesma amostra de 300 a 500 comentários: acurácia, F1, tempo e custo por mil avaliações, e análise dos erros de cada um |
| Dashboard | Motivos de reclamação por estado e por mês (Tableau Public ou Streamlit) |

## Stack

Python · pandas · scikit-learn · Jupyter · Gemini (saída estruturada) · Tableau Public

## Roadmap

- [x] Estrutura do repositório e plano do projeto
- [ ] Preparação e análise exploratória dos dados
- [ ] Modelo clássico (TF-IDF + regressão logística)
- [ ] Classificação com LLM e validação do JSON
- [ ] Tabela comparativa e análise de erros
- [ ] Classificação de motivos e dashboard
- [ ] Notebook publicado no Kaggle

## Resultados

_Serão publicados aqui ao final da comparação._

## Estrutura

```
data/        instruções para baixar o dataset (os CSVs não vão para o Git)
notebooks/   análise exploratória, modelo clássico, LLM e comparação
src/         funções reutilizadas pelos notebooks (carga de dados, prompts)
resultados/  tabelas e gráficos gerados
docs/        imagens usadas neste README
```

---

Feito por **Matteus Prado** · [LinkedIn](https://www.linkedin.com/in/matteus-prado-31b18a364) · [GitHub](https://github.com/MatGIT19)
