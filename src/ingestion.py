from typing import Optional, Union
from pathlib import Path
import datetime
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.ingestion import IngestionPipeline


from src.models import llm_factory, embedder_factory
from src.utils import graph_store_factory
from src.prompts import DEFAULT_KG_TRIPLET_EXTRACT_TMPL
from src.settings import SYSTEM_SETTINGS


# Pipeline of transformations to apply to the text data before indexing.
DEFAULT_INGESTION_PIPELINE = [
    TokenTextSplitter(
        chunk_size=SYSTEM_SETTINGS.chunk_size, 
        chunk_overlap=SYSTEM_SETTINGS.chunk_overlap
    ),
    SimpleLLMPathExtractor(
        llm=llm_factory(), 
        extract_prompt=DEFAULT_KG_TRIPLET_EXTRACT_TMPL, 
        num_workers=SYSTEM_SETTINGS.max_workers, 
        max_paths_per_chunk=SYSTEM_SETTINGS.max_paths_per_chunk,
    ),
    ImplicitPathExtractor(),
]

def get_meta(file_path):
    """
    Function for adding additional metadata to documents and chunks being ingested.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "directory": str(Path(file_path).parent),
        "file_name": str(Path(file_path).name),
        "created_at": now,
    }

class DocumentSettings:
    """
    A settings class for configuring the metadata and text templates for a Document instance.
    """
    def __init__(
            self,
            metadata_seperator: str = "\n",
            metadata_template: str = "{key}: {value}",
            text_template: str = "Metadata:\n{metadata_str}\n\nContent:\n{content}",
            excluded_embed_metadata_keys: list[str]=[],
            excluded_llm_meta_keys: list[str]=["directory", "file_name", "created_at"],
    ):
        """
        Initialize the DocumentSettings.

        Args:
            metadata_seperator: str - the separator to use when joining metadata key-value pairs.
            metadata_template: str - the template for formatting metadata key-value pairs.
            text_template: str - the template for formatting the metadata and content of the document.
            excluded_llm_meta_keys: list[str] - a list of metadata keys to exclude from the LLM when retrieving.
            excluded_embed_metadata_keys: list[str] - a list of metadata keys to exclude from the embeddings when indexing.
        """
        self.metadata_seperator = metadata_seperator
        self.metadata_template = metadata_template
        self.text_template = text_template
        self.excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self.excluded_llm_meta_keys = excluded_llm_meta_keys

    
    def configure_document(self, document:Document)->Document:
        """
        This method configures the metadata and text templates for a Document instance.

        Args:
            document: Document - the document instance to configure.

        Returns:
            Document - the configured document instance.
        """
        document.metadata_seperator = self.metadata_seperator
        document.metadata_template = self.metadata_template
        document.text_template = self.text_template
        if self.excluded_llm_meta_keys:
            document.excluded_llm_metadata_keys = self.excluded_llm_meta_keys
        if self.excluded_embed_metadata_keys:
            document.excluded_embed_metadata_keys = self.excluded_embed_metadata_keys
        return document

class DirectoryIngestionProcessor:
    """
    A processor class for ingesting text data from a directory and indexing it into a property graph.
    """
    def __init__(
            self,
            transformations: IngestionPipeline = DEFAULT_INGESTION_PIPELINE,
            llm_factory: callable = llm_factory,
            embedder_factory: callable = embedder_factory,
            metadata_callable: Union[callable, None] = get_meta,
            graph_store_factory: callable = graph_store_factory,
            document_settings: DocumentSettings = DocumentSettings(),
        ):
        """
        Initialize the DirectoryIngestionProcessor.

        Args:
            transformations: IngestionPipeline - the pipeline of transformations to apply to the text data before indexing.
            llm_factory: callable - a factory function for creating an LLM instance.
            embedder_factory: callable - a factory function for creating an Embedder instance.
            metadata_callable: callable - a function for adding additional metadata to documents and chunks being ingested.
            graph_store_factory: callable - a factory function for creating a property graph store instance.
            document_settings: DocumentSettings - the settings for configuring the metadata and text templates for a Document instance
        """
        self.transformations = transformations
        self.llm_factory = llm_factory
        self.embedder_factory = embedder_factory
        self.metadata_callable = metadata_callable
        self.graph_store_factory = graph_store_factory
        self.document_settings = document_settings

    def _read_directory(
            self, 
            directory: str, 
            recursive: bool = False,
            input_files: list[str] = None,
            exclude_files: list[str] = None,
            exclude_hidden_files: bool = True,
            errors: str = "ignore",
            num_files_limit: Optional[int] = None,
        )->list[Document]:
        reader = SimpleDirectoryReader(
            input_dir=directory,
            input_files=input_files,
            exclude=exclude_files,
            exclude_hidden=exclude_hidden_files,
            errors = errors,
            recursive=recursive,
            encoding="utf-8",
            filename_as_id=False,
            num_files_limit=num_files_limit,
            file_metadata=self.metadata_callable, 
            raise_on_error=False
        )
        documents = reader.load_data()
        for doc in documents:
            self.document_settings.configure_document(doc)
        return documents
    
    def process_directory(
            self,
            directory: str,
            recursive: bool = True,
            input_files: list[str] = None,
            exclude_files: list[str] = None,
            exclude_hidden_files: bool = True,
            errors: str = "ignore",
            num_files_limit: Optional[int] = None,
        )->None:
        """
        This method processes a directory of text files, ingesting the text data and indexing it into a property graph.

        Args:
            directory: str - the directory containing the text files to process.
            recursive: bool - whether to recursively search subdirectories.
            input_files: list[str] - a list of specific files to include.
            exclude_files: list[str] - a list of specific files to exclude.
            exclude_hidden_files: bool - whether to exclude hidden files.
            errors: str - how to handle errors during file reading.
            num_files_limit: Optional[int] - the maximum number of files to read.
        """
        documents = self._read_directory(
            directory=directory,
            recursive=recursive,
            input_files=input_files,
            exclude_files=exclude_files,
            exclude_hidden_files=exclude_hidden_files,
            errors=errors,
            num_files_limit=num_files_limit,
        )
        index = PropertyGraphIndex.from_documents(
            documents=documents,
            llm=self.llm_factory(),
            property_graph_store=self.graph_store_factory(),
            show_progress=False,
            embed_kg_nodes=True,
            transformations=self.transformations,
        )
