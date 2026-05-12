from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

from pydantic import Field, ConfigDict
from langchain_core.documents import Document
from typing import Any, List

from base_rag import BaseRAG
from utils.create_kb import initialize_kb
import torch
from pathlib import Path

device = ''

if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "utils" / "prompt.txt"

with open(PROMPT_PATH, 'r') as file:
    content = file.read()

class MyRAG(BaseRAG):
    def __init__(self, llm_instance, param1=None, param2=None, **kwargs):
        self.embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device}
        )
        self.vector_store, self.chunks = initialize_kb(embeddings_model=self.embeddings_model)
        self.bm25 = BM25Retriever.from_documents(self.chunks, k=5)
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        self.prompt = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template=content
        )
        self.retriever = self._build_retriever()
        super().__init__(llm_instance, **kwargs)

    # Retriever com MMR e EnsembleRetriever
    def _build_retriever(self, quantity: int = 5):
        semantic = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": quantity, "fetch_k": 30, "lambda_mult": 0.6}
        )

        ensemble = EnsembleRetriever(
            retrievers=[self.bm25, semantic],
            weights=[0.5, 0.5]
        )

        cross_encoder = self.cross_encoder

        # Cross-encoder -> Principal peça para o re-ranking
        class CrossEncoderRetriever(BaseRetriever):
            vectorstore: Any
            cross_encoder: Any
            k: int = Field(default=10)
            rerank_top_k: int = Field(default=3)

            model_config = ConfigDict(arbitrary_types_allowed=True)

            def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
                initial_docs = ensemble.invoke(query)
                pairs = [[query, doc.page_content] for doc in initial_docs]
                scores = self.cross_encoder.predict(pairs)
                scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

            async def _aget_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
                raise NotImplementedError("Async não implementado")

        return CrossEncoderRetriever(
            vectorstore=self.vector_store,
            cross_encoder=cross_encoder,
            k=quantity * 2,
            rerank_top_k=3
        )
    

    # Função específica pra rodar o RAG workflow
    def answer_question(self, question: str, mostrar_chunks: bool = False) -> str:
        docs = self.retriever.invoke(question)

        if mostrar_chunks:
            print("----")
            for i, doc in enumerate(docs):
                page = doc.metadata.get('page_display', 'p. ?')
                print(f" => chunk {i+1} [{page}]: {doc.page_content[:80]}...")
            print("----")

        contexto = "\n---\n".join(
        f"[{doc.metadata.get('page_display', 'p. ?')}] {doc.page_content}"
        for doc in docs
        )
        final_prompt = self.prompt.format(contexto=contexto, pergunta=question)

        response = self._generate_response(system_prompt='', user_prompt=final_prompt)

        # Caso seja interessante mostrar os chunks
        if mostrar_chunks:
            chunks_texto = "\n\n".join(
            f"[{doc.metadata.get('page_display', 'p. ?')}] {doc.page_content}" for doc in docs)
            return response, chunks_texto
        
        return response

    # Limpar a rodagem do RAG
    def teardown(self) -> None:
        self.vector_store._client.close()
        
        del self.embeddings_model
        del self.vector_store
        del self.chunks
        del self.bm25
        del self.cross_encoder
        del self.prompt
        del self.retriever
        del self.llm_instance

        print("Recursos de hardware liberados.")



# Exemplo de como instanciar um modelo do HuggingFace e instanciar o RAG

if __name__ == "__main__":

    MODEL_ID = "google/gemma-2-2b-it"

    local_llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text-generation",
        pipeline_kwargs=dict(
            do_sample=True,
            max_new_tokens=2048,
            return_full_text=False  # Atenção: Precisa setar este valor para contornar um bug!!!
        )
    )
    chat_model = ChatHuggingFace(llm=local_llm)

    rag = MyRAG(llm_instance=chat_model)

    question = """
    A trágica quebra institucional do golpe civil-militar de 1964 interrompeu a sobrevida política de JK, situação que atingiu o ápice com a dolorosa cassação do seu mandato de senador e suspensão severa de seus direitos políticos em 8 de junho de 1964. Como o autor descreve detalhadamente a mecânica de pressão política e militar que emparedou o governo de Humberto Castello Branco e forçou a punição sumária do ex-presidente?
    A) O presidente Castello Branco, motivado intimamente por um repúdio ideológico cultivado durante toda a sua carreira, exigiu e redigiu o decreto de expurgo de forma autoritária e despótica, isolando-se das deliberações conjuntas do Conselho de Segurança Nacional Agindo por vontade unânime própria, o general argumentou, baseando-se em convicções irrevogáveis e anunciando publicamente na TV, que Juscelino era o orquestrador financeiro da guerrilha comunista que tomou conta das praças de Belo Horizonte.
    B) O afastamento compulsório e dramático de Juscelino foi exigido de forma quase intempestiva e veemente pelo ministro da Guerra, o general Costa e Silva, atuando sob forte clamor dos oficiais da "linha dura" e lacerdistas, os quais consideravam o senador o pior e maior entrave oculto da Revolução em decorrência de seu formidável favoritismo e projeção para as futuras eleições presidenciais. Pressionado impiedosamente por esse flanco radical militar e confrontado por um dossiê calhamaço com suspeitas insatisfatórias de corrupção pessoal, o ponderado Castello Branco cedeu à contingência e executou burocraticamente a condenação.
    C) Após conversações diplomáticas madrugadoras travadas discretamente num requintado apartamento em Copacabana, Juscelino optou por solicitar oficialmente ao general Castello Branco a suspensão abrandada dos seus próprios direitos eleitorais. Ao prever um insustentável derramamento de sangue que atingiria a oposição de sua época, JK aceitou migrar num exílio protegido até Paris, exigindo como única moeda de troca para o abandono político o direito de seu PSD figurar oficialmente na estruturação legal e militar do novo governo instituído e assumir postos nos ministérios da Defesa.
    D) Uma vigorosa articulação originada dentro dos gabinetes do Supremo Tribunal Federal, sob influência das intensas e assustadoras passeatas que cobravam probidade no país, elaborou e aplicou uma decisão unânime limitadora do mandato e imunidade de Juscelino. O poder Judiciário considerou devidamente provadas todas as transações escusas do político mineiro com embaixadores ideológicos em Havana, coagindo moral e legalmente um presidente Castello Branco hesitante a endossar o afastamento do líder como uma determinação inquestionável e constitucional da Justiça pátria.
    """

    resposta, chunks = rag.answer_question(question, mostrar_chunks=True)
    print(resposta)
    print("---------")
    print(chunks)
    rag.teardown()
