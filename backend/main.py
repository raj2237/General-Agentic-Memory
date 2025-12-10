from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import chat, documents, memory
from core.gam_manager import gam_manager

app = FastAPI(title="GAM Chatbot with PDF Upload")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(memory.router)

@app.on_event("startup")
async def startup():
    await gam_manager.init()
    print("GAM Brain + PDF Support Loaded!")

@app.get("/")
def home():
    return {"message": "GAM Chatbot with PDF Upload READY!"}