from fastapi import FastAPI
from uvicorn import run as app_run

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == '__main__':
    app_run(app, host='0.0.0.0', port=8080)
