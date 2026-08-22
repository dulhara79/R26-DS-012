from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .models import AskRequest, DocumentStatus, SyncRequest
from .runtime import Runtime, build_runtime
from .util import redact_sensitive_settings


class ReviewRequest(BaseModel):
    reason: str = Field(default="manual_review", min_length=2, max_length=1000)


def create_app(runtime: Runtime | None = None, settings: Settings | None = None) -> FastAPI:
    supplied_runtime = runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = supplied_runtime or build_runtime(settings)
        yield

    app = FastAPI(
        title="CARE-AnxRAG",
        version="0.1.0",
        description="Contradiction-, authority-, reliability-, and evidence-aware anxiety RAG.",
        lifespan=lifespan,
    )

    def get_runtime() -> Runtime:
        return app.state.runtime

    def require_admin(
        x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
        rt: Runtime = Depends(get_runtime),
    ) -> None:
        if not rt.settings.admin_key:
            raise HTTPException(
                status_code=503,
                detail="Administrative API is disabled until CARE_ADMIN_KEY is configured",
            )
        if x_admin_key is None or not secrets.compare_digest(
            x_admin_key, rt.settings.admin_key
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing admin key")

    def require_admin_for_debug(
        request: AskRequest,
        x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
        rt: Runtime = Depends(get_runtime),
    ) -> None:
        if not request.include_debug:
            return
        if not rt.settings.admin_key:
            raise HTTPException(
                status_code=503,
                detail="Debug retrieval is disabled until CARE_ADMIN_KEY is configured",
            )
        if x_admin_key is None or not secrets.compare_digest(
            x_admin_key, rt.settings.admin_key
        ):
            raise HTTPException(status_code=401, detail="Debug retrieval requires an admin key")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return _HOME_HTML

    @app.get("/health")
    def health(rt: Runtime = Depends(get_runtime)) -> Any:
        status = rt.health()
        return {
            "status": status.status,
            "database": status.database,
            "vector_store": status.vector_store,
            "ollama": status.ollama,
        }

    @app.post("/v1/ask", dependencies=[Depends(require_admin_for_debug)])
    def ask(request: AskRequest, rt: Runtime = Depends(get_runtime)) -> Any:
        return rt.rag.answer(request.question, include_debug=request.include_debug)

    @app.post("/v1/retrieve", dependencies=[Depends(require_admin)])
    def retrieve(request: AskRequest, rt: Runtime = Depends(get_runtime)) -> Any:
        return rt.retriever.retrieve(request.question)

    @app.post("/v1/sync", dependencies=[Depends(require_admin)])
    def sync(request: SyncRequest, rt: Runtime = Depends(get_runtime)) -> Any:
        return rt.ingestion.sync(
            source_ids=request.source_ids or None,
            dry_run=request.dry_run,
            force=request.force,
        )

    @app.get("/v1/stats")
    def stats(rt: Runtime = Depends(get_runtime)) -> Any:
        return rt.database.stats()

    @app.get("/v1/sources", dependencies=[Depends(require_admin)])
    def sources(rt: Runtime = Depends(get_runtime)) -> Any:
        output = []
        states = {source.id: rt.database.get_source_state(source.id) for source in rt.sources}
        for source in rt.sources:
            safe_settings = redact_sensitive_settings(source.settings)
            output.append(
                {
                    **source.model_dump(mode="json"),
                    "settings": safe_settings,
                    "state": states[source.id].model_dump(mode="json"),
                }
            )
        return output

    @app.get("/v1/review/staging", dependencies=[Depends(require_admin)])
    def staging(rt: Runtime = Depends(get_runtime)) -> Any:
        return rt.database.list_staging_versions()

    @app.post("/v1/review/{version_id}/approve", dependencies=[Depends(require_admin)])
    def approve(version_id: str, rt: Runtime = Depends(get_runtime)) -> Any:
        try:
            rt.ingestion.approve(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"version_id": version_id, "status": DocumentStatus.ACTIVE.value}

    @app.post("/v1/review/{version_id}/reject", dependencies=[Depends(require_admin)])
    def reject(
        version_id: str,
        request: ReviewRequest,
        rt: Runtime = Depends(get_runtime),
    ) -> Any:
        try:
            rt.ingestion.reject(version_id, request.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"version_id": version_id, "status": DocumentStatus.REJECTED.value}

    @app.post("/v1/review/{version_id}/withdraw", dependencies=[Depends(require_admin)])
    def withdraw(
        version_id: str,
        request: ReviewRequest,
        rt: Runtime = Depends(get_runtime),
    ) -> Any:
        try:
            withdrawn_version_id = rt.ingestion.withdraw(version_id, request.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "requested_version_id": version_id,
            "withdrawn_version_id": withdrawn_version_id,
            "status": DocumentStatus.WITHDRAWN.value,
        }

    @app.post("/v1/reconcile", dependencies=[Depends(require_admin)])
    def reconcile(
        reset_embedding_index: bool = False,
        rt: Runtime = Depends(get_runtime),
    ) -> Any:
        return rt.ingestion.reconcile_active_vectors(
            reset_embedding_identity=reset_embedding_index
        )

    return app


_HOME_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CARE-AnxRAG</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}
textarea{width:100%;min-height:110px;padding:12px;font:inherit}button{padding:10px 18px;margin-top:10px}
pre{white-space:pre-wrap;background:#f4f4f4;padding:16px;border-radius:8px}.meta{font-size:.9rem;color:#555}
</style>
</head>
<body>
<h1>CARE-AnxRAG</h1>
<p>Evidence-grounded anxiety information. This research system is not a diagnostic or emergency service.</p>
<textarea id="question" placeholder="Ask an anxiety-information question..."></textarea><br>
<button onclick="ask()">Ask</button>
<pre id="answer">Ready.</pre>
<div id="sources"></div>
<script>
async function ask(){
 const answer=document.getElementById('answer'); const sources=document.getElementById('sources');
 answer.textContent='Searching evidence...'; sources.innerHTML='';
 try{
  const r=await fetch('/v1/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('question').value})});
  const data=await r.json(); if(!r.ok) throw new Error(JSON.stringify(data));
  answer.textContent=data.answer+'\n\nConfidence: '+data.confidence.toFixed(3)+' | Conflict: '+data.conflict_score.toFixed(3);
  if(data.citations?.length){
   const heading=document.createElement('h2'); heading.textContent='Sources'; sources.appendChild(heading);
   for(const c of data.citations){
    const p=document.createElement('p'); const strong=document.createElement('strong');
    strong.textContent=`[${c.citation_id}] ${c.title}`; p.appendChild(strong);
    p.appendChild(document.createElement('br'));
    p.appendChild(document.createTextNode(`${c.source_name} - ${c.evidence_level}`));
    if(c.url){try{const u=new URL(c.url); if(['http:','https:'].includes(u.protocol)){
      p.appendChild(document.createElement('br')); const a=document.createElement('a');
      a.href=u.href; a.target='_blank'; a.rel='noopener noreferrer'; a.textContent='Open source'; p.appendChild(a);
    }}catch(_){}}
    sources.appendChild(p);
   }
  }
 }catch(e){answer.textContent='Error: '+e.message;}
}
</script>
</body></html>"""


app = create_app()
