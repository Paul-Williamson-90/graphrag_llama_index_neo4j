from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from src.query import QueryEngineFactory
from src.prompts import DEFAULT_CHAT_PROMPT_TEMPLATE

class ChatEngineFactory:

    def __init__(
            self,
            query_engine_factory: QueryEngineFactory = QueryEngineFactory,
            prompt_template: PromptTemplate = DEFAULT_CHAT_PROMPT_TEMPLATE,
    ):
        self.query_engine_factory = query_engine_factory
        self.prompt_template = prompt_template

    def create(self):
        query_engine = self.query_engine_factory().create()
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=self.prompt_template,
            chat_history=[],
            verbose=True,
        )
        return chat_engine