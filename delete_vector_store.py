import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStoreDeleter:
    def __init__(self, vector_store_id: str = None):
        self.client = OpenAI()
        self.vector_store_id = vector_store_id
        self.base_dir = Path("build")
        self.id_file_path = self.base_dir / "vector_store_id.txt"

    def get_vector_store_id(self):
        """Get vector store ID from file or command line argument."""
        # If vector store ID is provided, use it
        if self.vector_store_id:
            return self.vector_store_id
        
        # Try to get vector store ID from file
        if self.id_file_path.exists():
            with open(self.id_file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        # If no ID is found, return None
        return None

    def delete_vector_store(self, vector_store_id: str = None):
        """Delete a vector store by ID."""
        # Get vector store ID
        if vector_store_id:
            self.vector_store_id = vector_store_id
        else:
            self.vector_store_id = self.get_vector_store_id()
        
        if not self.vector_store_id:
            print("No vector store ID provided or found in file.")
            return False
        
        try:
            # Get vector store details before deletion
            vector_store = self.client.vector_stores.retrieve(self.vector_store_id)
            print(f"\nVector Store: {vector_store.name}")
            print(f"ID: {vector_store.id}")
            
            # Confirm deletion
            confirm = input(f"\nAre you sure you want to delete the vector store '{vector_store.name}' (ID: {vector_store.id})? (yes/no): ")
            if confirm.lower() != "yes":
                print("Deletion cancelled.")
                return False
            
            # Delete the vector store
            self.client.vector_stores.delete(self.vector_store_id)
            print(f"Successfully deleted vector store: {vector_store.name}")
            
            # Remove the ID file if it exists
            if self.id_file_path.exists():
                self.id_file_path.unlink()
                print(f"Removed vector store ID file: {self.id_file_path}")
            
            return True
            
        except Exception as e:
            print(f"Error deleting vector store: {str(e)}")
            return False

def main():
    # Get vector store ID from command line argument if provided
    vector_store_id = None
    if len(sys.argv) > 1:
        vector_store_id = sys.argv[1]
    
    # Create deleter instance
    deleter = VectorStoreDeleter(vector_store_id)
    
    # Delete the vector store
    deleter.delete_vector_store()

if __name__ == "__main__":
    main() 