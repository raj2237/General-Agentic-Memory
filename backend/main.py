from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import chat, documents, memory
from core.gam_manager import gam_manager

app = FastAPI(title="GAM Chatbot with PDF Upload")

# CORS Configuration - Order matters!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# Include routers AFTER middleware
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(memory.router)

@app.on_event("startup")
async def startup():
    await gam_manager.init()
    print("🚀 Server ready!")

@app.get("/")
def home():
    return {"message": "GAM Chatbot with PDF Upload READY!"}

@app.get("/health")
def health_check():
    """Health check endpoint for testing"""
    return {
        "status": "healthy",
        "message": "Server is running",
        "gam_initialized": gam_manager.initialized
    }