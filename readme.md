# GraphRAG with Llama-Index, OpenAI, and Neo4J Graph Database

## Resources
- https://docs.llamaindex.ai/en/stable/

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Ensure Docker Desktop is running
3. Build the Neo4J container by running in the CLI:
```bash
make build
```
4. In your browser, go to: bolt://localhost:7687
- Login initially using:
    - username: neo4j
    - password: neo4j
- You will need to create a new password now, for the sakes of ease, I used:
    - username: neo4j
    - password: testing12345678
5. Create a .env file in the root of the project folder, you will need the following env vars:
```env
NEO_USERNAME=...
NEO_PASSWORD=...
NEO_URL=bolt://localhost:7687
NEO_DATABASE=...

OPENAI_API_KEY=...
```
6. demo_ingest.py has a quick script for ingesting the demo documents found in example_data/
```bash
python demo_ingest.py
```
7. Now the documents in the example data directory are ingested, you can view them in your Neo4j instance.
- Go to bolt://localhost:7687
- Run the following in the instance to see a pretty graph:
```neo4j
match (n)-[r]-(m) return n, r, m
```
8. You can also interact with the data using an LLM, this is called GraphRAG, where RAG = Retrieval Augmented Generation
- See demo_interact.ipynb for details on how to do this

## General 'gist' of how it works
### Ingestion (src.ingestion.py)
- Documents are first 'chunked' into smaller pieces of text (chunks). This is to improve the specificity of generated vector embeddings that represent the text, and also for ensuring text size going into the LLM do not explode / exceed the context window limit.
- SimpleLLMPathExtractor is a process that for each text chunk, will use an LLM to identify entities, objects, and the relationships between them, in the style of (entity)->(predicates)->(object)
    - Entities and objects becomes nodes on the graph, whereas the predicates are the relationships between the nodes
    - Chunks also have their own node on the graph
- ImplicitPathExtractor seeks to find relationships between existing nodes but not necessarily within the same text chunk
- The chunks, identified entities and relationships, and any additional metadata are added to the knowledge graph, along with vector embeddings generated for all.
### Retrieval & answer (src.query.py)
- A user inputs a prompt into the system, normally a question regarding the data within the graph database.
- LLMSynonymRetriever will use an LLM to generate N many keywords from the user's prompt which are used to search the graph (entities, relationships, objects), return a number of chunks that match
- VectorContextRetriever will use vector similarity search between the user prompt and embedding vectors contained in the graph
- PGRetriever combines both retrievers into one workflow
- A response synthesiser is simply the process the LLM will follow to reduce the found chunks into a single answer that is passed back to the user
### Stateful retrieval & answer (src.chat.py)
- The Q&A feature can be extended into a chat LLM agent, where previous prompts and responses are saved into a list and added to the new prompt as a chat history for additional context to the LLM.