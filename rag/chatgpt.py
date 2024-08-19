from typing import Iterable
from openai import OpenAI
from dotenv import load_dotenv
import os

class ChatGPT:
    def __init__(self) -> None:
        API_KEY_PATH = "api_keys.env"
        load_dotenv(API_KEY_PATH)
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)


    def chat(self, message: str, history:list[dict], rag_context: str = ""):
        conversation=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. Use any relavant information provided in the context to give a personalised response based on the context if provided. In the final json return the answer under 'response'. Don't include anything else."}
        ]
        for h in history:
            conversation.append(h) 

        query = f"Context: {rag_context}. Question: {message}"
        conversation.append({"role": "user", "content": query})


        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        messages=conversation) # type: ignore

        return response.choices[0].message.content




