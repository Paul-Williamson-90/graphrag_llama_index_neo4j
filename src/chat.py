from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from src.query import QueryEngineFactory
from src.prompts import DEFAULT_CHAT_PROMPT_TEMPLATE

class ChatEngineFactory:
    """
    Factory for creating a ChatEngine instance.
    The ChatEngine instance allows a user to send questions to the LLM and DB and receive responses,
    with the history of requests and responses stored in the chat engine state and used as additional
    context with the LLM when facilitating conversation with the user.
    """
    def __init__(
            self,
            query_engine_factory: QueryEngineFactory = QueryEngineFactory,
            prompt_template: PromptTemplate = DEFAULT_CHAT_PROMPT_TEMPLATE,
    ):
        """
        Initialize the ChatEngineFactory.

        Args:
            query_engine_factory: QueryEngineFactory - the factory for creating the QueryEngine instance.
            prompt_template: PromptTemplate - the template for the prompt to send to the LLM.
        """
        self.query_engine_factory = query_engine_factory
        self.prompt_template = prompt_template

    def create(self, verbose: bool = True, **kwargs)->CondenseQuestionChatEngine:
        """
        Create a ChatEngine instance.

        Args:
            verbose: bool - whether to print debug information.
            kwargs: dict - additional keyword arguments to pass to the query engine factory.

        Returns:
            CondenseQuestionChatEngine - the chat engine instance.
        """
        query_engine = self.query_engine_factory(verbose=verbose, **kwargs).create()
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=self.prompt_template,
            chat_history=[],
            verbose=verbose,
        )
        return chat_engine