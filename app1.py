from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import sqlite3
from diffusers import DiffusionPipeline
import torch

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app = Flask(__name__)

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries. If you don't know the answer, say don't know "),
        ("user","Question:{question}")
    ]
)

## LLAMA2 LLm
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

## Stable Diffusion for image generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form['input_text']
    response = chain.invoke({"question":input_text})
    
    # Check if the user wants an image response
    if "generate image" in input_text.lower():
        image_path = generate_image(input_text)
        response += f" Image: <img src='{image_path}'>"
    
    save_chat_history(input_text, response)
    return jsonify({'response': response})

def generate_image(prompt):
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    image_path = f"images/{prompt}.png"
    image.save(image_path)
    return image_path

def save_chat_history(input_text, response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (input_text TEXT, response TEXT)''')
    c.execute("INSERT INTO chat_history VALUES (?,?)", (input_text, response))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)