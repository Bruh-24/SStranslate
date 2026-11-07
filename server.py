from fastapi import FastAPI

server = FastAPI()

@server.get("/health-check")
def health():
     return {"status": "OK"}