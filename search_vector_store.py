import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStoreSearcher:
    def __init__(self, vector_store_id: str):
        self.client = OpenAI()
        self.vector_store_id = vector_store_id

    def get_file_chunks(self, file_id: str, file_name: str):
        """Get all chunks for a specific file."""
        try:
            print(f"\nRetrieving chunks for file: {file_name}")
            print(f"File ID: {file_id}")
            print("-" * 50)
            
            # Search for chunks from this specific file
            chunks = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=f"file:{file_id}"  # Search for chunks from this specific file
            )
            
            # Print all chunks from this file
            print(f"Found {len(chunks.data)} chunks:")
            for i, chunk in enumerate(chunks.data, 1):
                print(f"\nChunk {i}:")
                print(chunk.content)
                print("-" * 30)
            
            print("=" * 50)
            
        except Exception as e:
            print(f"Error retrieving chunks for file {file_name}: {str(e)}")

    def list_all_files_and_chunks(self):
        """List all files and their chunks in the vector store."""
        try:
            # Get vector store details
            vector_store = self.client.vector_stores.retrieve(self.vector_store_id)
            print(f"\nVector Store: {vector_store.name}")
            print(f"ID: {vector_store.id}")
            print("=" * 50)
            
            # Get all files in the vector store
            files = self.client.vector_stores.files.list(
                vector_store_id=self.vector_store_id
            )
            
            print(f"\nFound {len(files.data)} files in the vector store")
            
            # For each file, get its metadata and chunks
            for file in files.data:
                try:
                    file_details = self.client.files.retrieve(file.id)
                    self.get_file_chunks(file.id, file_details.filename)
                except Exception as file_error:
                    print(f"Error retrieving file details: {str(file_error)}")
        except Exception as e:
            print(f"Error listing files and chunks: {str(e)}")

    def search_vector_store(self, query: str):
        """Search the vector store with the given query."""
        try:
            # Get vector store details
            vector_store = self.client.vector_stores.retrieve(self.vector_store_id)
            print(f"\nVector Store: {vector_store.name}")
            print(f"ID: {vector_store.id}")
            print("=" * 50)
            
            print(f"\nSearching for: {query}")
            print("-" * 50)
            
            # Search the vector store
            search_results = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query
            )
            
            # Sort results by score in descending order and get top 3
            sorted_results = sorted(search_results.data, key=lambda x: x.score, reverse=True)[:3]
            
            # Print top 3 search results
            print(f"\nTop 3 most relevant results:")
            for i, result in enumerate(sorted_results, 1):
                print(f"\nResult {i}:")
                print(f"Score: {result.score:.4f}")
                print("-" * 30)
                print(f"Content: {result.content}")
                print("=" * 50)
                
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")

def main():
    # Vector store ID from the build process
    vector_store_id = "vs_67efb9d45e9481919f37a906a91b2c0a"
    
    # Create searcher instance
    searcher = VectorStoreSearcher(vector_store_id)
    
    # Get query from command line argument or prompt for input
    if len(sys.argv) > 1:
        # Use command line argument as query
        query = " ".join(sys.argv[1:])
    else:
        # Prompt for query
        query = input("Enter your search query: ")
    
    # Search the vector store
    searcher.search_vector_store(query)

if __name__ == "__main__":
    main() 