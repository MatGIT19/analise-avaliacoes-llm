# Dados

Os arquivos do dataset não ficam no repositório. Para baixar os três usados pelo projeto (cerca de 40 MB):

```bash
python scripts/baixar_dados.py
```

O script usa a [cópia pública mantida pela Olist no GitHub](https://github.com/olist/work-at-olist-data). Também dá para baixar pelo [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e copiar para esta pasta:

- `olist_order_reviews_dataset.csv`: nota de 1 a 5 e comentário de cada avaliação
- `olist_orders_dataset.csv`: datas de compra, entrega real e entrega prevista de cada pedido
- `olist_customers_dataset.csv`: estado e cidade de cada cliente

Licença do dataset: CC BY-NC-SA 4.0 (uso não comercial, com atribuição à Olist).
