from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

# Routers import
from rec_system.router import router as rec_router

# app init & router
app = FastAPI()

app.include_router(rec_router)


@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}


@app.get("/ready")
def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "Service is ready"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Please try again later."},
    )
