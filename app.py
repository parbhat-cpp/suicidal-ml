import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object,preprocess_text,avg_word2vec
from src.models import PredictionRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def force_https_urls(request: Request, call_next):
    # Railway uses x-forwarded-proto to indicate original protocol
    if request.headers.get("x-forwarded-proto") == "https":
        request.scope["scheme"] = "https"
    response = await call_next(request)
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

word2vec_model = load_object('./models/word2vec.pkl')
classifier_model = load_object('./models/model.pkl')

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse(request, 'index.html')

@app.post("/predict-mental-health")
async def predict(request: Request, pred_body: PredictionRequest):
    try:
        text = pred_body.text
        
        text = preprocess_text(text)
        text = avg_word2vec(word2vec_model, text)
        
        prediction = classifier_model.predict(np.array(text).reshape(1,-1))
        
        return JSONResponse(content={'prediction': prediction[0]})
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app_run(app, host='0.0.0.0', port=8080)
