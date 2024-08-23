
# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files, extract text from them, and interact with a chatbot that uses the retrieved information as context to provide relevant answers. The system is built using Python, Flask for the backend, and Streamlit for the frontend.

## Getting Started

### Prerequisites

- Python 3.10
- Git
- Docker and Docker Compose (if using Docker)
- OpenAI API Key (for ChatGPT integration)

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/aaronkher/rag_chatbot.git
cd rag_chatbot
```

#### 2. Create and activate a virtual environment

It is recommended to use a specific name for the virtual environment to keep the project organized.

```bash
python3 -m venv rag_chatbot_venv
source rag_chatbot_venv/bin/activate  # On Windows use `rag_chatbot_venv\Scripts\activate`
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers
python -m nltk.downloader punkt
```

#### 4. Set up the OpenAI API Key

You need to create a file named `api_keys.env` in the root directory of the project with the following content:

```bash
touch api_keys.env
```
Add the following line to `api_keys.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key. The environment variable must be named exactly `OPENAI_API_KEY` for the application to work correctly.

### Running the Application

#### 1. Run the API

```bash
python main/main_api.py
```

#### 2. Run the Frontend

In a new terminal window:

```bash
streamlit run frontend/chat_page.py
```

### Running with Docker

#### 1. Build the Docker Images

```bash
docker compose build
```

![Docker Build Screenshot](docs/images/docker_compose_build.png)

This command will build the Docker images for both the API and the frontend.

#### 2. Run the Services

```bash
docker compose up
```

![Docker Compose Up Screenshot](docs/images/docker_compose_up.png)

🚨 **Important:** It takes some time for both the API (main_api.py) and the frontend (chat_page.py) services to start up completely. Please wait until you see both the API and frontend are running before attempting to use the application. The API will be available at `http://localhost:8000` and the Frontend will be available at `http://localhost:8501`.

#### 3. Stop the services

```bash
docker compose down
```

### Project Structure

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

### Video Demonstration

Check out this [video demonstration](https://youtu.be/F9gwzJouI0w) that showcases the project and explains how RAG works.

### References

- **OCR Model (Nougat)**: [Hugging Face Nougat Documentation](https://huggingface.co/docs/transformers/en/model_doc/nougat)
- **Sentence Transformers for Vectorization**: [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
- **ChatGPT OpenAI LLM**: [OpenAI Chat Completions Guide](https://platform.openai.com/docs/guides/chat-completions)

Make sure to create the `api_keys.env` file as described above and correctly set the `OPENAI_API_KEY` variable.

