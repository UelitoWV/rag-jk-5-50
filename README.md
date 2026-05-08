[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YwAhie6h)
# RAG - 5 anos em 50

## Descrição

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** especializado na biografia de Juscelino Kubitschek, utilizando técnicas avançadas de recuperação híbrida e re-ranking para responder a perguntas complexas de múltipla escolha com alta precisão.

> Projeto desenvolvido para a disciplina de **Sistemas Multiagentes**, semestre 2026.1, BCC, UFRPE.

> Link do RAG via Colab: https://colab.research.google.com/drive/1YFrZ0bWEVApZeiTOSxwGsaH0kLYRA7_d?usp=sharing

## Equipe   

- Bruno de Melo Costa
- Wellington Viana da Silva Junior

## Técnicas de RAG Utilizadas

O sistema utiliza uma pipeline sofisticada para garantir a relevância e diversidade dos contextos recuperados:

- **Chunking Strategy**: Utiliza o `SemanticChunker` da LangChain, que divide o documento baseado na variação semântica entre sentenças (threshold de percentil 70), garantindo que cada fragmento mantenha uma unidade de significado coesa.
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, um modelo multilíngue eficiente para representação vetorial de sentenças.
- **Vector Store**: **ChromaDB**, utilizado para persistência local dos embeddings e busca vetorial.
- **Retrieval Strategy**: **Busca Híbrida (Ensemble)** combinando:
    - **BM25**: Recuperação baseada em palavras-chave, ideal para nomes próprios e termos específicos.
    - **MMR (Maximal Marginal Relevance)**: Busca semântica que penaliza a redundância, selecionando resultados relevantes e diversos.
- **Reranking**: Implementado via `CrossEncoder` com o modelo `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. O re-ranking refina os resultados do ensemble inicial, reordenando os chunks de acordo com a relevância direta à pergunta.
- **Prompt Engineering**: Prompt estruturado para análise rigorosa de alternativas de múltipla escolha, forçando o modelo a justificar a veracidade de cada opção baseando-se estritamente no contexto.

## Estrutura do Projeto

```
├── README.md
├── requirements.txt                   # Dependências do projeto
├── chroma_db/                         # Base de vetores persistida
├── kb/
│   └── raw/                           # Documento PDF (biografia de JK)
├── notebooks/                         # Experimentos utilizados como teste para a aplicação do RAG final
└── src/                               # Código-fonte principal
    ├── base_rag.py                    # Interface abstrata para o sistema RAG
    ├── rag.py                         # Implementação da classe MyRAG e pipeline principal
    ├── RAG_5_anos_em_50.py            # Notebook implementado via Colab
    └── utils/
        └── create_kb.py               # Scripts para processamento de PDF e criação da base
```

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/UFRPE-SMA-2026-1/projeto-1-rag-rag-5-anos-em-50.git
   cd projeto-1-rag-rag-5-anos-em-50
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate   
   # Windows
   .venv\Scripts\activate      
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. (Opcional) Configure seu ambiente para uso de GPU (CUDA) se disponível, o código detectará automaticamente via `torch.cuda.is_available()`.

## Como Executar

O ponto de entrada principal é o arquivo `src/rag.py`. Ele contém um exemplo de execução utilizando o modelo `google/gemma-2b-it`.

Para rodar o sistema:

```bash
cd src
python rag.py
```

O script irá:
1. Inicializar a base de conhecimento (carregando do PDF ou do disco se já existir).
2. Carregar o modelo de linguagem via HuggingFace.
3. Processar uma pergunta de exemplo e exibir a resposta estruturada com justificativas.

## Licença

Este projeto está sob a licença MIT.
