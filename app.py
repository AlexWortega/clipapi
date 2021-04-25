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

def trans(translator,text):
  trans = translator.translate(str(text),dest='en')
  return trans.text

translator = Translator()

images=[]



#TEXT=trans(translator,data[i])

from big_sleep import Imagine
def gen(TEXT,ite):
  """
  Text входная строка по английски
  return

  """
  SAVE_EVERY =  1
  SAVE_PROGRESS = False 
  LEARNING_RATE = 5e-2 

  

    
  

  
  
  images=[]
  (model) = Imagine(
    text = TEXT,
    save_every = SAVE_EVERY,
    lr = LEARNING_RATE,
    iterations = 2,
    save_progress = SAVE_PROGRESS,
    seed = 0
  )
  for epoch in range(1):#20

      for i in range(ite):#1000
        (model.train_step(epoch, i))#
        images+=[ Image(f'{[i]}.png')]

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import FileResponse

import cv2
app = FastAPI()



@app.get("/")
async def main():
    return "go to /text2image/?it=10?message=какая то русская строка"



@app.get('/text2image/{it}')
async def detect_spam_query(it: int,message: str):
  text = trans(translator,message)
  gen(text,it)
  print('sending')
  return FileResponse(text.replace(' ','_')+'.png')







import nest_asyncio
from pyngrok import ngrok
import uvicorn


nest_asyncio.apply()
uvicorn.run(app, port=80,timeout_keep_alive=100)
