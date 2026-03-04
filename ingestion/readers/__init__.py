"""ingestion/readers — universal source adapters"""
from ingestion.readers.file_reader import FileReader
from ingestion.readers.api_reader import APIReader
from ingestion.readers.db_reader import DBReader
from ingestion.readers.stream_reader import StreamReader

__all__ = ["FileReader", "APIReader", "DBReader", "StreamReader"]
