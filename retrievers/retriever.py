from typing import Any
from abc import ABC, abstractmethod

class Retriever(ABC):
    """
    Abstract base class for Retriever implementations.
    This class provides a common structure for different types of retrievers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_retriever(self, api_key) -> Any:
        """
        Abstract method for retrieving knowledge base.

        Returns:
            Any: knowledge base.
        """
        pass