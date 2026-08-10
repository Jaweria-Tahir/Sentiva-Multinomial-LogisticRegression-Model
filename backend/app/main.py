from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict


app = FastAPI(
    title="Sentiva Sentiment API"
)

origins = [
    "https://sentiva-multinomial-logistic-regres.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict.router)
@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiva API is running"}