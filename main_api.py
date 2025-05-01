from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.project_api import router as project_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(project_router)

@app.get("/")
def read_root():
    return {"message": "Nexora API is running 🚀"}
