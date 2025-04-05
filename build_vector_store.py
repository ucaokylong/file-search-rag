import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStoreBuilder:
    def __init__(self):
        self.client = OpenAI()
        self.base_dir = Path("build")
        self.datas_dir = self.base_dir / "datas"
        self.create_build_directory()

    def create_build_directory(self) -> None:
        """Create the build directory structure and initialize it with necessary files."""
        # Create the build directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)
        
        # Create the datas directory
        self.datas_dir.mkdir(exist_ok=True)
        
        # Create a README.md file in the build directory
        build_readme_path = self.base_dir / "README.md"
        if not build_readme_path.exists():
            with open(build_readme_path, "w", encoding="utf-8") as f:
                f.write("# Build Directory\n\n")
                f.write("This directory contains all build-related files and data.\n\n")
                f.write("## Structure\n\n")
                f.write("- `datas/`: Contains files to be processed for the vector store\n")
                f.write("- `vector_store_id.txt`: Stores the ID of the created vector store\n")
        
        # Create a README.md file in the datas directory
        datas_readme_path = self.datas_dir / "README.md"
        if not datas_readme_path.exists():
            with open(datas_readme_path, "w", encoding="utf-8") as f:
                f.write("# Data Directory\n\n")
                f.write("This directory contains files that will be processed and added to the vector store.\n\n")
                f.write("## Supported File Types\n\n")
                f.write("- Text files (.txt)\n")
                f.write("- PDF documents (.pdf)\n")
                f.write("- Markdown files (.md)\n")
                f.write("- Word documents (.doc, .docx)\n")
                f.write("- HTML files (.html)\n")
                f.write("- JSON files (.json)\n")
                f.write("- Programming language files (.py, .js, .java, .cpp, .cs, .go, .rb, .php, .css, .sh, .tex, .ts)\n\n")
                f.write("Place your files in this directory and run the build_vector_store.py script to process them.\n")
        
        print(f"Build directory structure initialized at: {self.base_dir.absolute()}")

    def get_supported_files(self) -> List[Path]:
        """Get all supported files from the datas directory."""
        supported_extensions = {
            '.txt', '.pdf', '.md', '.doc', '.docx', '.html', '.json',
            '.py', '.js', '.java', '.cpp', '.cs', '.go', '.rb', '.php',
            '.css', '.sh', '.tex', '.ts'
        }
        
        files = []
        if self.datas_dir.exists():
            for file_path in self.datas_dir.rglob("*"):
                if file_path.suffix.lower() in supported_extensions:
                    files.append(file_path)
        return files

    def create_vector_store(self, name: str = "Document Store") -> str:
        """Create a vector store and return its ID."""
        vector_store = self.client.vector_stores.create(name=name)
        return vector_store.id

    def save_vector_store_id(self, vector_store_id: str) -> None:
        """Save the vector store ID to a file."""
        id_file_path = self.base_dir / "vector_store_id.txt"
        with open(id_file_path, "w", encoding="utf-8") as f:
            f.write(vector_store_id)
        print(f"Vector store ID saved to: {id_file_path}")

    def upload_files_to_vector_store(self, vector_store_id: str) -> None:
        """Upload files from datas directory to the vector store."""
        files = self.get_supported_files()
        if not files:
            print("No supported files found in the datas directory.")
            return

        # Create file streams for upload
        file_streams = [open(file_path, "rb") for file_path in files]
        
        try:
            # Upload files and poll for completion
            file_batch = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=file_streams
            )
            
            print(f"Upload status: {file_batch.status}")
            print(f"File counts: {file_batch.file_counts}")
            
        finally:
            # Close all file streams
            for stream in file_streams:
                stream.close()

    def build(self) -> str:
        """Build the vector store and return its ID."""
        # Create vector store
        vector_store_id = self.create_vector_store()
        print(f"Created vector store with ID: {vector_store_id}")
        
        # Upload files
        self.upload_files_to_vector_store(vector_store_id)
        
        # Save the vector store ID
        self.save_vector_store_id(vector_store_id)
        
        return vector_store_id

def main():
    builder = VectorStoreBuilder()
    vector_store_id = builder.build()
    print(f"\nVector store building complete. ID: {vector_store_id}")
    print("You can use this ID to attach the vector store to your assistant.")

if __name__ == "__main__":
    main() 