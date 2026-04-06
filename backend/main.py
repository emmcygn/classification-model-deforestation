from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.pipeline import router as pipeline_router
from api.routes.explorer import router as explorer_router

app = FastAPI(title="DeforestAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.include_router(pipeline_router)
app.include_router(explorer_router)
