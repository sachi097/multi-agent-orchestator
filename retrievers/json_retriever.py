from retrievers import Retriever
from phi.knowledge.json import JSONKnowledgeBase
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
import os

class JSONRetriever(Retriever):
    def __init__(self):
        super().__init__()

    def get_retriever(self, api_key: str):
        vector_db = LanceDb(
            table_name="questions",
            uri="/tmp/lancedb",
            search_type=SearchType.keyword,
            embedder=OpenAIEmbedder(model="text-embedding-3-small", api_key=api_key)
        )

        knowledge_base = JSONKnowledgeBase(
            path=os.path.dirname(__file__) + "/data.json",
            vector_db=vector_db,
        )

        knowledge_base.load(recreate=False)

        return knowledge_base
        