from math import e
from openai import OpenAI
import numpy as np
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    while True:
        try:
            emb = client.embeddings.create(input = [text], model=model).data[0].embedding
            emb_array = emb
            return emb_array
        except Exception as e:
            print(f"llm call error: {e}\n retry!")
            continue
    