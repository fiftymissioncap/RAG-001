"""
This module provides a simple in-memory data store and a retriever class for searching documents using LLamaIndex.

It includes the following classes:
- MemoryDataStore: A basic in-memory key-value store for documents.
- LLamaIndexRetriever: A retriever that uses LLamaIndex to search documents stored in a MemoryDataStore.
"""

from llama_index import LLamaIndex

class MemoryDataStore:
    """
    A simple in-memory data store to store and retrieve documents.
    """

    def __init__(self):
        """
        Initializes the MemoryDataStore by creating an empty dictionary to store documents.
        """
        self.store = {}

    def add_document(self, key, document):
        """
        Add a document to the data store.

        Args:
            key (str): The key to associate with the document.
            document (str): The content of the document.
        """
        self.store[key] = document

    def get_document(self, key):
        """
        Retrieve a document from the data store.

        Args:
            key (str): The key associated with the desired document.

        Returns:
            str: The document content if found, otherwise None.
        """
        return self.store.get(key, None)

class LLamaIndexRetriever:
    """
    A retriever class that uses LLamaIndex to search for documents.

    This class bridges the gap between a data store and the LLamaIndex search
    capabilities. It retrieves a document from the provided data store and then
    uses the LLamaIndex to perform a search on its content.
    """

    def __init__(self, index: LLamaIndex, data_store: MemoryDataStore):
        """
        Initialize the LLamaIndexRetriever.

        Args:
            index (LLamaIndex): An instance of LLamaIndex to perform searches.
            data_store (MemoryDataStore): An instance of MemoryDataStore to store and retrieve documents.
        """
        self.index = index
        self.data_store = data_store

    def retrieve(self, key):
        """
        Retrieve and search for a document using LLamaIndex.

        Args:
            key (str): The key associated with the desired document.

        Returns:
            list: The search result from LLamaIndex if the document is found, otherwise None.
        """
        document = self.data_store.get_document(key)
        if document is not None:
            result = self.index.search(document)
            return result
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Initialize the memory data store and add some documents
    data_store = MemoryDataStore()
    data_store.add_document("doc1", "This is the content of document 1.")
    data_store.add_document("doc2", "This is the content of document 2.")

    # Initialize the LLamaIndex
    llama_index = LLamaIndex()

    # Create the retriever
    retriever = LLamaIndexRetriever(llama_index, data_store)

    # Retrieve and search for a document
    search_result = retriever.retrieve("doc1")
    print(search_result)
