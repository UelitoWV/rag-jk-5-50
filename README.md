[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YwAhie6h)
# [Nome da Equipe / Projeto]

## Descrição

Descreva brevemente o objetivo do projeto e o problema que ele resolve.

> Projeto desenvolvido para a disciplina de **Sistemas Multiagentes**, semestre 2026.1, BCC, UFRPE.

## Equipe

- Bruno de Melo Costa
- Wellington Viana da Silva Junior

## Técnicas de RAG Utilizadas

Liste e descreva as técnicas de RAG (Retrieval-Augmented Generation) adotadas pela equipe, por exemplo:

- **Chunking Strategy**: descreva como os documentos são divididos (tamanho do chunk, sobreposição, etc.).
- **Embedding Model**: modelo utilizado para gerar os embeddings (ex.: `text-embedding-ada-002`, `sentence-transformers/all-MiniLM-L6-v2`).
- **Vector Store**: banco de vetores utilizado (ex.: FAISS, Chroma, Pinecone).
- **Retrieval Strategy**: estratégia de recuperação (ex.: similaridade de cosseno, BM25 híbrido, MMR).
- **Reranking**: se aplicável, descreva o método de reranking utilizado.
- **Outras técnicas**: ex.: HyDE, query expansion, self-query retriever, etc.


## Estrutura do Projeto

```
├── README.md
├── src/                # Código-fonte principal
├── kb/
│   └── raw/            # Documento bruto usado da base de conhecimento
├── notebooks/          # Notebooks de experimentos e análises
├── requirements.txt    # Dependências do projeto
└── .github/
    └── pull_request_template.md
```


## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
   cd <seu-repositorio>
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente ...


## Como Executar

...


## Licença

Este projeto está sob a licença MIT.
