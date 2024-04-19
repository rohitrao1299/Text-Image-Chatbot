from flask import Flask, render_template, request, jsonify, send_file
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import sqlite3
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import io

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

# Function to generate AI-based images using Stable Diffusion
def generate_images_using_stable_diffusion(text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    prompt = text
    image = pipe(prompt).images[0]
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form['input_text']
    response = chain.invoke({"question":input_text})
    save_chat_history(input_text, response)
    return jsonify({'response': response})

def save_chat_history(input_text, response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (input_text TEXT, response TEXT)''')
    c.execute("INSERT INTO chat_history VALUES (?,?)", (input_text, response))
    conn.commit()
    conn.close()

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        text = request.form['text']
        image_output = generate_images_using_stable_diffusion(text)
        img_io = io.BytesIO()
        image_output.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, attachment_filename='generated_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)