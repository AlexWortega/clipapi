from tqdm.notebook import trange
from IPython.display import Image, display
import os
from big_sleep import Imagine
from random import randint


from googletrans import Translator

def trans(translator,text):
  trans = translator.translate(str(text),dest='en')
  return trans.text

translator = Translator()

images=[]



#TEXT=trans(translator,data[i])

from big_sleep import Imagine

def gen(TEXT):
  """
  Text входная строка по английски
  return

  """
  SAVE_EVERY =  1
  SAVE_PROGRESS = False 
  LEARNING_RATE = 5e-2 

  

    
  

  
  
  images=[]
  model = Imagine(
    text = TEXT,
    save_every = SAVE_EVERY,
    lr = LEARNING_RATE,
    iterations = 2,
    save_progress = SAVE_PROGRESS,
    seed = randint(0,1000)
  )
  for epoch in trange(2, desc = 'epochs'):#20

      for i in trange(20, desc = 'iteration'):#1000
        model.train_step(epoch, i)
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
    return "api"



@app.get('/text2image/')
async def detect_spam_query(message: str):
  text = trans(translator,message)
  gen(text)
  
  return FileResponse(text+'.png')




import nest_asyncio
from pyngrok import ngrok
import uvicorn


uvicorn.run(app,timeout_keep_alive=100)