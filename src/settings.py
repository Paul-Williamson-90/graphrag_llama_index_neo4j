from pydantic import BaseModel

class SystemSettings(BaseModel):

    max_workers: int = 4
    use_async: bool = True
    chunk_size: int = 300
    chunk_overlap: int = 100
    max_paths_per_chunk: int = 10
    max_tokens: int = 1000
    model: str = "gpt-4o"    

SYSTEM_SETTINGS = SystemSettings()