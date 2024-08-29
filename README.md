
# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files, extract text from them, and interact with a chatbot that uses the retrieved information as context to provide relevant answers. The system is built using Python, Flask for the backend, and Streamlit for the frontend.

## Getting Started

### Prerequisites

- Python 3.10
- Git
- Docker and Docker Compose (if using Docker)
- `kind` (Kubernetes in Docker)
- OpenAI API Key (for ChatGPT integration)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/aaronkher/rag_chatbot.git
   cd rag_chatbot
   ```

2. **Create and activate a virtual environment**

   It is recommended to use a specific name for the virtual environment to keep the project organized.

   ```bash
   python3 -m venv rag_chatbot_venv
   source rag_chatbot_venv/bin/activate  # On Windows use `rag_chatbot_venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers
   python -m nltk.downloader punkt
   ```

4. **Set up the OpenAI API Key**

   You need to set the OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Running the Application

### Important Configuration Note
In the Streamlit frontend code (`frontend/chat_page.py`), ensure the `BACKEND_API` variable points to the correct backend service URL:

- For **Local Environment (without Docker)** setup, uncomment the following line and comment out the Docker line:
  ```python
  # BACKEND_API = os.getenv('BACKEND_API', 'http://localhost:8000')
  ```

- For **Docker** setup, use:
  ```python
  BACKEND_API = os.getenv('BACKEND_API', 'http://api:8000')
  ```

- For **Kubernetes** setup, uncomment the following line and comment out the Docker line:
  ```python
  # BACKEND_API = os.getenv('BACKEND_API', 'http://my-api-service:8000')
  ```

**Note**: The `BACKEND_API` is set for Docker functionality by default. Make sure to update it accordingly if you are using Kubernetes or running the application locally without Docker

### Option 1: Local Environment

1. **Run the API**

   ```bash
   python main/main_api.py
   ```

2. **Run the Frontend**

   In a new terminal window:

   ```bash
   streamlit run frontend/chat_page.py
   ```

### Option 2: Docker Compose

1. **Build the Docker Images**

   ```bash
   docker compose build
   ```

   This command will build the Docker images for both the API and the frontend.

2. **Run the Services**

   ```bash
   docker compose up
   ```

   The API will be available at `http://localhost:8000` and the Frontend will be available at `http://localhost:8501`.

   🚨 **Important:** It takes some time for both the API (main_api.py) and the frontend (chat_page.py) services to start up completely. Please wait until you see both the API and frontend are running before attempting to use the application. 


3. **Stop the services**

   ```bash
   docker compose down
   ```

### Option 3: Kubernetes with `kind`

1. **Create a `kind` Cluster**

   ```bash
   kind create cluster --name rag-chatbot-cluster
   ```

2. **Load Docker Images into `kind`**

   ```bash
   kind load docker-image my-api:latest --name rag-chatbot-cluster
   kind load docker-image my-frontend:latest --name rag-chatbot-cluster
   ```

3. **Deploy the Application to Kubernetes**

   Apply the deployment and service YAML files:

   ```bash
   kubectl apply -f api-deployment.yaml
   kubectl apply -f frontend-deployment.yaml
   ```

4. **Port Forwarding to Access the Services**

   Run these commands in separate terminals:

   ```bash
   kubectl port-forward service/my-api-service 30001:8000
   kubectl port-forward service/my-frontend-service 30002:8501
   ```

   The API will be accessible at `http://localhost:30001` and the frontend at `http://localhost:30002`.

### Stopping Services and Cluster

#### For Kubernetes (`kind`)

1. **Stop Port Forwarding:**

   Press `Ctrl + C` in the terminal where port forwarding is running.

2. **Delete the Kubernetes Deployment and Services:**

   ```bash
   kubectl delete deployment my-api my-frontend
   kubectl delete service my-api-service my-frontend-service
   ```

3. **Delete the `kind` Cluster:**

   ```bash
   kind delete cluster --name rag-chatbot-cluster
   ```

#### For Docker Compose

1. **Stop the Services:**

   ```bash
   docker compose down
   ```

#### For Local Environment (Python Virtual Environment)

1. **Deactivate Virtual Environment:**

   ```bash
   deactivate
   ```

2. **Stop Running Services:**

   Press `Ctrl + C` in the terminal where each service is running.

## Project Structure

- `rag/`: Contains the main backend logic.
  - `chatgpt.py`: Handles interactions with the ChatGPT model.
  - `document_vectorizer.py`: Vectorizes documents for retrieval.
  - `ocr.py`: Handles OCR tasks for PDF text extraction.
- `main/`: Contains the entry point for the Flask API.
  - `main_api.py`: The Flask API service.
- `frontend/`: Contains the Streamlit frontend application.
  - `chat_page.py`: The Streamlit frontend service.
- `retriever/`: Contains the logic for retrieving relevant document chunks.
- `text_documents/`: Stores the text extracted from the PDFs.
- `Dockerfile.api`: Dockerfile for the API service.
- `Dockerfile.frontend`: Dockerfile for the Frontend service.
- `docker-compose.yml`: Docker Compose configuration for running the project.
- `api-deployment.yaml`: Kubernetes Deployment configuration for the API service.
- `api-service.yaml`: Kubernetes Service configuration for exposing the API service.
- `frontend-deployment.yaml`: Kubernetes Deployment configuration for the frontend service.
- `frontend-service.yaml`: Kubernetes Service configuration for exposing the frontend service.
- `kind-config.yaml`: Configuration file for creating a Kubernetes cluster using `kind`.
- `api_keys.env`: Environment file containing sensitive information such as the OpenAI API key.

### Video Demonstration

Check out this [video demonstration](https://youtu.be/F9gwzJouI0w) that showcases the project and explains how RAG works.

### References

- **OCR Model (Nougat)**: [Hugging Face Nougat Documentation](https://huggingface.co/docs/transformers/en/model_doc/nougat)
- **Sentence Transformers for Vectorization**: [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
- **ChatGPT OpenAI LLM**: [OpenAI Chat Completions Guide](https://platform.openai.com/docs/guides/chat-completions)

Make sure to create the `api_keys.env` file as described above and correctly set the `OPENAI_API_KEY` variable.
