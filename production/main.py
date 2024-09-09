import os
from dotenv import load_dotenv

from huggingface_hub import login
login(token=os.getenv("READ_HF"))

from datetime import datetime
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, logging, pipeline,BitsAndBytesConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import re, pandas as pd
import ast, uvicorn

class Llama():
    
    def load_model(self):
        model_name = "tiiqunetwork/llama2-7b-qnal5-1"
        bnb_4bit_compute_dtype = "float16"
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        print(f"Starting to load the model {model_name} into memory"+ str(datetime.now()))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.bos_token_id = 1
        self.stop_token_ids = [0]
        
        print("End: "+ str(datetime.now()))
    
    def getres(self, long_string):
        # pattern = r'\[.*?\{.*?\}.*?\]'
        pattern = re.compile("'\{\'.*?\'\}'")
        matches = re.findall(pattern, long_string)
        n_match = len(matches)
        dict_list = dict()
        results = str(n_match) + " >"
        for dict_list_str in matches:
            try:
                # dict_list = ast.literal_eval(dict_list_str)
                results = results + str(dict_list_str)
    #             if 'Question' in dict_list:
    #                 results = results + "Question: " + dict_list['Question'] + "\n"
    #             if 'Answer' in dict_list:
    #                 results = results + "Answer: " + dict_list['Answer'] + "\n"
    #             if 'Topic' in dict_list:
    #                 results = results + "Topic: " + dict_list['Topic'] + "\n"
            except Exception as e:
                results = results + f"Error: {e}" + "\n"
        return results
    
    def calculate_max_length(self, input_text):
        return int(len(input_text.split()) * 2.6)
    
    def resval(self, parr):
        topic_list = [
                "Analysis", "Science and tech", "Factual", "Management", "Taxonomy", 
                "Strategy", "Ethics and regulation"]
        prompt = f"""<s>[INST]Complete the following tasks from the review text: 
        - Generate two questions using between 5 and 20 words
        - Generate one answer per question using between 5 and 20 words
        - Choose one topic to label the generated question and answer from a list of topics: {", ".join(topic_list)}
        The review is delimited with triple backticks. \
        Format your response as a JSON object with \
        "Question", "Answer" and "Topic" as the keys.
        Make sure to avoid hallucination for the question and answer.
        The question and answer should contain only words from the review text.
        Review text: '''{parr}'''""".strip() + "[/INST]"
        custom_max = self.calculate_max_length(prompt)
        # print(prompt)
        self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tok, max_length=custom_max)
        result = self.pipe(prompt)
        self.model_out = result[0]['generated_text']
        return self.getres(self.model_out)

sentiment_model = Llama()    

class TextInput(BaseModel):
    inputs: str
    parameters: dict[str, Any] | None

app = FastAPI()
# load the model asynchronously on startup
# Old FastAPI
@app.on_event("startup")
async def startup():
    sentiment_model.load_model()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     sentiment_model.load_model()

# middlewares
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

@app.get('/')
async def root():
    return {'hello': 'world'}

@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        return {"generated_text": sentiment_model.resval(data.inputs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

uvicorn.run(app)
