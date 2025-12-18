import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import gradio as gr

# -------------------
# LLM
# -------------------
OLLAMA = "gpt-oss:20b"
OLLAMA_API = "http://localhost:11434/api/chat"

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")













system_prompt = """
You are an expert in cheminformatics. You receive questions such as:
“What is the melting point of CCO?”

From each question, you must:

Identify the SMILES string.

Identify the requested molecular property.

Resolve the SMILES to the correct compound and retrieve the requested property only from the specified source. Report the value using standard units and clearly state the source.

If the SMILES is invalid, ambiguous, or the property is not available from the requested source, state this clearly. Do not infer or use other sources.
"""

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = ollama.chat.completions.create(model="gemini-2.5-flash", messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    gr.ChatInterface(chat).launch()