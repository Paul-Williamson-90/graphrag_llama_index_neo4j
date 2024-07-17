from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

from src.models import llm_factory
from src.utils import graph_store_factory
from src.prompts import DEFAULT_SYNONYM_EXTRACT_TEMPLATE_TMPL


class QueryEngineFactory:

    def __init__(
            self,
            llm_factory: callable = llm_factory,
            graph_store_factory: callable = graph_store_factory,
            use_async: bool = False,
            verbose: bool = True,
            response_mode: str = "tree_summarize",
            num_workers: int = 4,
            max_keywords: int = 10,
            top_k_similar: int = 5,
            path_depth: int = 1,
    ):
        self.llm_factory = llm_factory
        self.graph_store_factory = graph_store_factory
        self.use_async = use_async
        self.verbose = verbose
        self.response_mode = response_mode
        self.num_workers = num_workers
        self.max_keywords = max_keywords
        self.top_k_similar = top_k_similar
        self.path_depth = path_depth

    def _get_retriever(self):
        graph_store = self.graph_store_factory()
        retriever_list = [
            LLMSynonymRetriever(
                graph_store=graph_store,
                include_text=True,
                synonym_prompt=DEFAULT_SYNONYM_EXTRACT_TEMPLATE_TMPL,
                max_keywords=self.max_keywords,
            ),
            VectorContextRetriever(
                graph_store=graph_store,
                include_text=True,
                similarity_top_k=self.top_k_similar,
                path_depth=self.path_depth,
            ),
        ]
        retriever = PGRetriever(
            sub_retrievers=retriever_list,
            num_workers=self.num_workers,
            use_async=self.use_async,
            show_progress=self.verbose,
        )
        return retriever
    
    def _get_response_synthesizer(self):
        synth = get_response_synthesizer(
            llm=self.llm_factory(),
            use_async=self.use_async,
            verbose=self.verbose,
            response_mode=self.response_mode,
        )
        return synth
    
    def create(self):
        retriever = self._get_retriever()
        response_synthesizer = self._get_response_synthesizer()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine