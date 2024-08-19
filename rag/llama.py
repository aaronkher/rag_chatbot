import os
import llama_cpp
from typing import List, Dict, Union

class Llama:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            chat_format="llama-2",
            verbose=True
        )
        print("-------")
    
    def chat(self, message: str, history: List[Dict[str, str]], rag_context: str = "") -> str:
        conversation: List[Dict[str, Union[str, dict]]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant designed to output JSON. "
                    "Use any relevant information provided in the context to give a personalized response. "
                    "In the final JSON return the answer under 'response'. Don't include anything else."
                )
            }
        ]
        conversation.extend(history)
        query = f"Context: {rag_context}. Question: {message}"
        conversation.append({"role": "user", "content": query})
        response = self.model.create_chat_completion(
            messages=conversation
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
        else:
            content = '{"response": "No response from the assistant."}'

        return content

if __name__ == "__main__":
    local_model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    model_path = local_model_path

    llama = Llama(model_path=model_path)
    query = "What is 1+1?"
    response_obj = llama.chat(query, history=[], rag_context="")
    print(response_obj)
