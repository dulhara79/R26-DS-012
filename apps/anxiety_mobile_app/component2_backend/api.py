from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException

try:
    from .reporting_processor import Component2Processor
except ImportError:
    from reporting_processor import Component2Processor


app = FastAPI(title="Aura Component 2 API", version="1.0.0")


async def get_processor() -> Component2Processor:
    return Component2Processor.from_env()


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Supabase bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Supabase bearer token.")
    return token


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "timezone": os.getenv("COMPONENT2_TIMEZONE", "Asia/Colombo")}


@app.get("/behavioral/{participant_code}")
async def behavioral(
    participant_code: str,
    authorization: str | None = Header(default=None),
    processor: Component2Processor = Depends(get_processor),
) -> dict:
    token = _bearer_token(authorization)
    try:
        if not await processor.verify_participant_token(participant_code, token):
            raise HTTPException(status_code=403, detail="Participant identity mismatch.")
        return await processor.behavioral_payload(participant_code)
    finally:
        await processor.db.close()


@app.post("/internal/process/{participant_code}")
async def process_one(
    participant_code: str,
    x_processor_key: str | None = Header(default=None),
    processor: Component2Processor = Depends(get_processor),
) -> dict:
    expected = os.getenv("COMPONENT2_PROCESSOR_KEY", "")
    if not expected or x_processor_key != expected:
        await processor.db.close()
        raise HTTPException(status_code=401, detail="Invalid processor key.")
    try:
        return await processor.process_participant(participant_code)
    finally:
        await processor.db.close()


@app.post("/internal/process-all")
async def process_all(
    x_processor_key: str | None = Header(default=None),
    processor: Component2Processor = Depends(get_processor),
) -> list[dict]:
    expected = os.getenv("COMPONENT2_PROCESSOR_KEY", "")
    if not expected or x_processor_key != expected:
        await processor.db.close()
        raise HTTPException(status_code=401, detail="Invalid processor key.")
    try:
        return await processor.process_all()
    finally:
        await processor.db.close()
