import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from rag.chatgpt import ChatGPT  
from rag.document_vectorizer import DocumentVectorizer
from retriever.retriever import Retriever
from rag.ocr import OCR

# print("Python version:", sys.version)
# print("Current working directory:", os.getcwd())
# print("Contents of current directory:", os.listdir())
# print("All modules imported successfully")

app = Flask(__name__)

print("API Running.")

conversation_history = [] # history
combined_df = pd.DataFrame()  # in-memory vectorized data (vector db)
ocr = OCR()
model = SentenceTransformer("all-MiniLM-L6-v2")
RETRIEVER_SIMILARITY_THRESHOLD = 0.1 # for retriever 

# uploaded document location
TEXT_DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), '../text_documents')

# model_path = '/Users/aaronkher/Documents/VScodeProjects/retrieval-augmented-generation/models/llama-2-7b-chat.Q4_K_M.gguf'
# llama = Llama(model_path=model_path)

@app.route('/')
def health_check():
    return "API is running", 200

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global combined_df
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.pdf'):
        pdf_path = os.path.join('/tmp', file.filename)
        file.save(pdf_path)
        
        text = ocr.extract_text_from_pdf(pdf_path)
        
        if text:
            txt_filename = file.filename.replace('.pdf', '.txt')
            txt_path = os.path.join(TEXT_DOCUMENTS_FOLDER, txt_filename)

            with open(txt_path, 'w') as txt_file:
                txt_file.write(text)
            
            vectorizer = DocumentVectorizer(embedding_model=model)
            df = vectorizer.create_dataframe(txt_path)
            
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            return jsonify({"message": "File processed and vectorized in memory successfully"}), 200
        else:
            return jsonify({"error": "Failed to extract text from the PDF"}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global combined_df, conversation_history
    message = request.json.get('message')
    similarity = None
    if message:
        if combined_df is not None and not combined_df.empty:
            retriever = Retriever()
            query_embedding = retriever.get_sentence_embedding(sentence=message)
            top_k_chunks = retriever.get_top_k_similar_chunks(combined_df, query_embedding, k=2)

            context_retrieved = ""
            for chunk in top_k_chunks:
                text, similarity = chunk[0], chunk[1]
                if similarity > RETRIEVER_SIMILARITY_THRESHOLD:
                    context_retrieved += text
        else:
            context_retrieved = ""

        # # when using Llama
        # response_message = llama.chat(
        #     message=message, 
        #     history=conversation_history, 
        #     rag_context=context_retrieved
        # )

        chatgpt = ChatGPT()
        response_message = chatgpt.chat(
            message=message, 
            history=conversation_history, 
            rag_context=context_retrieved
        )

        print("--::RETRIEVED TEXT::--")
        print(context_retrieved)
        print("-")
        print(f"[similarity: {similarity}]")
        print("--::--::--::--")

        if response_message:
            response_message = str(response_message)
        else:
            response_message = '{"response": "No response from the assistant."}'

        response_text = json.loads(response_message)["response"]
        
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response_text})

        return jsonify({
            "response": response_text,
            "history": conversation_history
        }), 200
    
    return jsonify({"error": "No message sent"}), 400


# # local
# if __name__ == '__main__':
#     if not os.path.exists(TEXT_DOCUMENTS_FOLDER):
#         os.makedirs(TEXT_DOCUMENTS_FOLDER)
    
#     app.run(port=8000, debug=True)

# docker
if __name__ == '__main__':
    print("Starting Flask app...")
    if not os.path.exists(TEXT_DOCUMENTS_FOLDER):
        os.makedirs(TEXT_DOCUMENTS_FOLDER)
    #print(f"TEXT_DOCUMENTS_FOLDER created: {os.path.exists(TEXT_DOCUMENTS_FOLDER)}")
    
    try:
        app.run(host='0.0.0.0', port=8000)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        import traceback
        traceback.print_exc()
