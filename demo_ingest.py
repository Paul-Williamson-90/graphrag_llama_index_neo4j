from src.ingestion import DirectoryIngestionProcessor

if __name__=="__main__":
    processor = DirectoryIngestionProcessor()
    processor.process_directory(directory="example_data")