import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, vector_store_id: str):
        self.client = OpenAI()
        self.vector_store_id = vector_store_id
        self.conversation_history = []
        
        # Create an assistant with file search capability
        self.assistant = self.client.beta.assistants.create(
            name="RAG Chatbot",
            instructions="You answer questions based on the files provided in the vector store.",
            model="gpt-3.5-turbo",
            tools=[{"type": "file_search"}]
        )
        
        # Create a thread for the conversation
        self.thread = self.client.beta.threads.create()
        print(f"Created assistant with ID: {self.assistant.id}")
        print(f"Created thread with ID: {self.thread.id}")
        
        # Get files from vector store
        self.setup_files()

    def setup_files(self):
        """Get files from vector store and attach them to the assistant."""
        try:
            # Get vector store details
            vector_store = self.client.vector_stores.retrieve(self.vector_store_id)
            print(f"\nVector Store: {vector_store.name}")
            print(f"ID: {vector_store.id}")
            
            # Get all files in the vector store
            files = self.client.vector_stores.files.list(
                vector_store_id=self.vector_store_id
            )
            
            print(f"\nFound {len(files.data)} files in the vector store")
            
            # Collect file IDs
            file_ids = []
            
            # Get file details and collect IDs
            for file in files.data:
                try:
                    # Get file details
                    file_details = self.client.files.retrieve(file.id)
                    print(f"Found file: {file_details.filename}")
                    file_ids.append(file.id)
                except Exception as file_error:
                    print(f"Error getting file details: {str(file_error)}")
            
            # Update assistant with all file IDs at once
            if file_ids:
                try:
                    self.client.beta.assistants.update(
                        assistant_id=self.assistant.id,
                        file_ids=file_ids
                    )
                    print(f"Successfully attached {len(file_ids)} files to the assistant")
                except Exception as update_error:
                    print(f"Error updating assistant with files: {str(update_error)}")
            else:
                print("No files were found to attach")
                    
        except Exception as e:
            print(f"Error setting up files: {str(e)}")

    def generate_response(self, user_input: str):
        """Generate a response using the Assistants API."""
        try:
            # Add the user's message to the thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_input
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )
            
            # Wait for the run to complete
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    return "Sorry, there was an error processing your request."
            
            # Get the assistant's response
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            
            # Get the latest assistant message
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
            
            return "Sorry, I couldn't generate a response."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def start_chat(self):
        """Start an interactive chat session."""
        print("\nWelcome to the RAG-powered Chatbot!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("=" * 50)
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! Have a great day!")
                break
            
            # Generate and print response
            response = self.generate_response(user_input)
            print("\nAssistant:", response)
            print("-" * 50)

def main():
    # Vector store ID from the build process
    vector_store_id = "vs_67efb9d45e9481919f37a906a91b2c0a"
    
    # Create chatbot instance
    chatbot = RAGChatbot(vector_store_id)
    
    # Start chat session
    chatbot.start_chat()

if __name__ == "__main__":
    main() 