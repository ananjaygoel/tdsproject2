from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os
import time
from agent import run_agent   # this function will do the actual quiz-solving

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME)}

@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    # Step 1: Validate JSON
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if "secret" not in data or "url" not in data or "email" not in data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Step 2: Validate secret
    if data["secret"] != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    quiz_url = data["url"]

    # Step 3: Start background worker to solve quiz
    background_tasks.add_task(run_agent, quiz_url, data["email"], data["secret"])

    # Step 4: Immediately return 200
    return JSONResponse({"status": "ok"}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
