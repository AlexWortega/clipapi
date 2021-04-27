from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import FileResponse

import cv2
app = FastAPI()

from tqdm.notebook import trange
from IPython.display import Image, display
import os
from big_sleep import Imagine
from random import randint
import torch

from googletrans import Translator

import numpy as np
from random import randint
import os
import base64


import cv2
from PIL import Image

from big_sleep import Imagine
from models import SRPredictor






"""
-------------------------MODEL--------------------
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def translate_rus2eng(text):
    translator = Translator()
    trans = translator.translate(str(text), dest='en')
    return trans.text

def encode_img(img):
    jpg_img = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(jpg_img[1]).decode('utf-8')
    return img_b64

def generate(text, iters):
    TEXT = text
    SAVE_EVERY = 100
    SAVE_PROGRESS = False 
    LEARNING_RATE = 5e-2
    ITERATIONS = 100
    SEED = randint(0,100)

    images=[]
    model = Imagine(
        text = TEXT,
        save_every = SAVE_EVERY,
        lr = LEARNING_RATE,
        iterations = ITERATIONS,
        save_progress = SAVE_PROGRESS,
        seed = SEED
      )
    
    epoch = 0
    for i in range(iters):
        model.train_step(epoch, i)
        
def text2image(text, iters):
    text = translate_rus2eng(text=text)
    generate(text=text, iters=iters)
    image = Image.open(text.replace(' ', '_')+'.png').convert('RGB')
    return image

def predict(request):
    text = str(request['text'])   
    iters = int(request['iters'])
    scale = int(request['scale'])
    
    sr = SRPredictor(device)
    sr.load_weights()
    image = text2image(text=text, iters=iters)
    result = sr.predict(image, scale=scale, decompress=False)
    return {"result": encode_img(np.array(result))}
  
  
"""
-------------------------API--------------------
"""  

app = FastAPI()



@app.get("/")
async def main():
    return "go to /text2image/?it=10?message=какая то русская строка"



@app.get('/text2image/{it}')
async def detect_spam_query(message: str,it: int,scale: int):
  
  req = {"text":message,"iters":it,"scale":scale}
  
  

  return predict(req)






import nest_asyncio
from pyngrok import ngrok
import uvicorn


nest_asyncio.apply()
uvicorn.run(app, port=80,timeout_keep_alive=100)
