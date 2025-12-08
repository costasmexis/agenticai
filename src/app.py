from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

MODEL = "llama3.2:3b"
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
reader = PdfReader("../data/Profile.pdf")

linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

name = "Konstantinos Mexis"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = ollama.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    gr.ChatInterface(chat).launch()