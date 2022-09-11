from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Please, go to the /classify endpoint - bexgboost."}
