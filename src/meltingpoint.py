import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import gradio as gr

load_dotenv(override=True) # override=True means any existing environment variables will be overwritten by values from the .env file.

ollama = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai")

system_prompt = "You are a professional computational chemist specializing in molecular property prediction. \
Your task is to predict the melting point (in Kelvin) of a molecule given its SMILES string. \
Input: The user will provide a single SMILES string. \
Output: You must return only one number, representing the predicted melting point in Kelvin, with no explanation, no units, and no additional text. \
Feel free to seach internet for the metling point of the compound. \
If you are unsure, output your best estimate as a number. Do not include comments, symbols, units, or text. "

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = ollama.chat.completions.create(model="gemini-2.5-flash", messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    gr.ChatInterface(chat).launch()