from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

from src.models import llm_factory
from src.utils import graph_store_factory
from src.prompts import DEFAULT_SYNONYM_EXTRACT_TEMPLATE_TMPL


class QueryEngineFactory:
    """
    Factory class for creating a QueryEngine instance.

    The QueryEngine instance allows a user to send a question to the LLM and DB and receive a response.
    The session is not stateful.
    """
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
        """
        Initialize the QueryEngineFactory.

        Args:
            llm_factory: callable - a factory function for creating an LLM instance.
            graph_store_factory: callable - a factory function for creating a graph store instance.
            use_async: bool - whether to use asynchronous processing.
            verbose: bool - whether to print debug information.
            response_mode: str - the mode for generating the response using the found contexts.
            num_workers: int - the number of workers to use for processing.
            max_keywords: int - the maximum number of keywords to extract from the user prompt.
            top_k_similar: int - the number of similar contexts to retrieve from vector similarity search.
            path_depth: int - the depth of the paths to traverse in the graph store.
        """
        self.llm_factory = llm_factory
        self.graph_store_factory = graph_store_factory
        self.use_async = use_async
        self.verbose = verbose
        self.response_mode = response_mode
        self.num_workers = num_workers
        self.max_keywords = max_keywords
        self.top_k_similar = top_k_similar
        self.path_depth = path_depth

    def _get_retriever(self)->PGRetriever:
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
    
    def _get_response_synthesizer(self)->BaseSynthesizer:
        synth = get_response_synthesizer(
            llm=self.llm_factory(),
            use_async=self.use_async,
            verbose=self.verbose,
            response_mode=self.response_mode,
        )
        return synth
    
    def create(self)->RetrieverQueryEngine:
        """
        This method creates a QueryEngine instance.

        Returns:
            RetrieverQueryEngine - the query engine instance.
        """
        retriever = self._get_retriever()
        response_synthesizer = self._get_response_synthesizer()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine