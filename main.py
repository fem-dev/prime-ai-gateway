import base64, hashlib, os, secrets, sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "data" / "gateway.db")))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-32B")
QWEN_URL = os.getenv("QWEN_API_URL", "https://router.huggingface.co/v1/chat/completions")
FLUX_URL = os.getenv("FLUX_API_URL", "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell")

app = FastAPI(title="Prime AI Gateway", version="1.0.0")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET or "dev-only-change-me",
                   same_site="strict", https_only=os.getenv("COOKIE_SECURE", "false").lower() == "true",
                   session_cookie="prime_admin_session", max_age=28800)

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def setup_check():
    if not ADMIN_PASSWORD or not SESSION_SECRET:
        raise HTTPException(503, "Set ADMIN_PASSWORD and SESSION_SECRET in the environment.")

def admin(request: Request):
    setup_check()
    if request.session.get("admin") is not True:
        raise HTTPException(401, "Please sign in.")

def init_db():
    with closing(connect()) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS api_keys(
            id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE, prefix TEXT NOT NULL,
            created_at TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)""")
        c.commit()

@app.on_event("startup")
def startup():
    init_db()

async def api_key(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer API key.")
    raw = authorization.split(" ", 1)[1].strip()
    if not raw.startswith("prime-"):
        raise HTTPException(401, "Invalid API key.")
    with closing(connect()) as c:
        row = c.execute("SELECT id FROM api_keys WHERE key_hash=? AND revoked=0", (digest(raw),)).fetchone()
    if row is None:
        raise HTTPException(401, "Invalid or revoked API key.")
    return row

class Login(BaseModel):
    password: str

class NewKey(BaseModel):
    label: str = Field(default="My API key", min_length=1, max_length=60)

class Chat(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False

class ImageRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=1)
    size: str | None = "1024x1024"
    response_format: str | None = "b64_json"

LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prime Gateway</title><style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(ellipse at 20% 10%,#1b2d56,transparent 45%),#090d18;color:#edf2ff;font:16px system-ui}.card{width:min(420px,92vw);padding:34px;border:1px solid #293858;border-radius:22px;background:#10182a}h1{margin:10px 0}.muted{color:#a5b2cb;line-height:1.6}.orb{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#79a7ff,#8c6bff)}label{display:block;margin:24px 0 8px}input{width:100%;padding:14px;border-radius:10px;border:1px solid #344565;background:#0b1221;color:white}button{width:100%;padding:14px;border:0;border-radius:10px;background:linear-gradient(90deg,#639bff,#9274ff);color:white;font-weight:800;margin-top:16px}#err{color:#ff9eaa}</style></head><body><main class="card"><div class="orb"></div><p>PRIME / AI GATEWAY</p><h1>Welcome back</h1><p class="muted">Sign in to manage your private API keys.</p><form id="f"><label>Admin password</label><input id="p" type="password" required autocomplete="current-password"><button>Sign in</button><p id="err"></p></form></main><script>f.onsubmit=async e=>{e.preventDefault();let r=await fetch('/admin/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:p.value})});if(r.ok)location.href='/dashboard';else err.textContent=(await r.json()).detail||'Sign in failed'}</script></body></html>"""

DASH = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prime Gateway Dashboard</title><style>
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 75% 0,#1b2d56,transparent 40%),#090d18;color:#edf2ff;font:16px system-ui}header{padding:22px max(22px,calc((100vw - 1000px)/2));border-bottom:1px solid #293858;display:flex;justify-content:space-between}.wrap{max-width:1000px;margin:42px auto;padding:0 22px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{padding:24px;border:1px solid #293858;border-radius:18px;background:#10182a}.wide{grid-column:1/-1}.muted{color:#a5b2cb;line-height:1.6}.pill{display:inline-block;padding:7px 11px;border:1px solid #344b72;border-radius:22px;color:#a9c4ff;margin:4px}input{width:100%;padding:12px;border-radius:10px;border:1px solid #344565;background:#0b1221;color:white;margin:10px 0}button{padding:12px 16px;border:0;border-radius:10px;background:linear-gradient(90deg,#639bff,#9274ff);color:white;font-weight:800;cursor:pointer}.secondary{background:#1b2942;border:1px solid #3b4e70}.key{padding:13px;background:#0b1221;border:1px solid #344565;border-radius:10px;overflow-wrap:anywhere;color:#b9d0ff}.notice{color:#ffcf8b;font-size:13px}.status{color:#a9c4ff;font-size:13px}li{color:#b8c5dc;line-height:1.8}@media(max-width:700px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}</style></head><body><header><b>PRIME / AI GATEWAY</b><button class="secondary" id="out">Sign out</button></header><main class="wrap"><p>CONTROL CENTER</p><h1>Your API gateway</h1><p class="muted">Generate a gateway key and download its connection details for your apps.</p><div class="grid">
<section class="card"><h2>Models</h2><span class="pill">Qwen3 · chat</span><span class="pill">FLUX.1 Schnell · images</span><p class="muted">Provider credentials stay on the server.</p></section>
<section class="card"><h2>Endpoints</h2><ul><li>POST /v1/chat/completions</li><li>POST /v1/images/generations</li><li>GET /v1/models</li></ul></section>
<section class="card wide"><h2>Create API key</h2><p class="muted">The full key is shown only once. Store the TXT file privately.</p><label>Key label</label><input id="label" maxlength="60" value="My API key"><button id="make">Generate key + download TXT</button> <span id="status" class="status"></span><div id="result" style="display:none;margin-top:16px"><p class="notice">Save this key now; it cannot be displayed again.</p><div class="key" id="key"></div></div></section>
<section class="card wide"><h2>Security</h2><ul><li>Use environment variables for the admin password and provider token.</li><li>Never commit real credentials or downloaded keys.</li><li>Enable HTTPS and set COOKIE_SECURE=true in production.</li><li>Use persistent storage for the SQLite database.</li></ul></section></div></main><script>
make.onclick=async()=>{make.disabled=true;status.textContent='Generating…';result.style.display='none';try{let r=await fetch('/admin/api-keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:label.value||'My API key'})});let d=await r.json();if(!r.ok)throw Error(d.detail||'Could not create key');key.textContent=d.api_key;result.style.display='block';let blob=new Blob(['api key : '+d.api_key+'\\nbase url : '+d.base_url+'\\n'],{type:'text/plain'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='prime-api-key.txt';a.click();URL.revokeObjectURL(a.href);status.textContent='Created and downloaded.'}catch(e){status.textContent=e.message}finally{make.disabled=false}};
out.onclick=async()=>{await fetch('/admin/logout',{method:'POST'});location.href='/'};
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(LOGIN)

@app.post("/admin/login")
def login(data: Login, request: Request):
    setup_check()
    if not secrets.compare_digest(data.password, ADMIN_PASSWORD):
        raise HTTPException(401, "Incorrect admin password.")
    request.session["admin"] = True
    return {"ok": True}

@app.post("/admin/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    admin(request)
    return HTMLResponse(DASH)

@app.post("/admin/api-keys")
def create_key(data: NewKey, request: Request):
    admin(request)
    raw = "prime-" + secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    with closing(connect()) as c:
        c.execute("INSERT INTO api_keys(label,key_hash,prefix,created_at) VALUES(?,?,?,?)",
                  (data.label.strip() or "My API key", digest(raw), raw[:14], now))
        c.commit()
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return {"api_key": raw, "base_url": base + "/v1"}

@app.get("/admin/api-keys")
def get_keys(request: Request):
    admin(request)
    with closing(connect()) as c:
        rows = c.execute("SELECT id,label,prefix,created_at,revoked FROM api_keys ORDER BY id DESC").fetchall()
    return {"data": [dict(x) for x in rows]}

@app.post("/admin/api-keys/{key_id}/revoke")
def revoke_key(key_id: int, request: Request):
    admin(request)
    with closing(connect()) as c:
        cur = c.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,))
        c.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "API key not found.")
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def models(_=Depends(api_key)):
    now = int(datetime.now(timezone.utc).timestamp())
    return {"object":"list","data":[
        {"id":"qwen3","object":"model","created":now,"owned_by":"prime"},
        {"id":"flux.1-schnell","object":"model","created":now,"owned_by":"prime"}]}

def provider_headers():
    if not HF_TOKEN:
        raise HTTPException(503, "The server administrator has not configured HF_TOKEN.")
    return {"Authorization":"Bearer " + HF_TOKEN}

@app.post("/v1/chat/completions")
async def chat(payload: Chat, _=Depends(api_key)):
    if payload.stream:
        raise HTTPException(400, "Streaming is not enabled; use stream=false.")
    if not payload.messages:
        raise HTTPException(400, "messages must contain at least one message.")
    body = payload.model_dump(exclude_none=True)
    body["model"] = QWEN_MODEL
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
            r = await client.post(QWEN_URL, headers=provider_headers(), json=body)
    except httpx.RequestError:
        raise HTTPException(502, "Could not reach the Qwen provider.")
    if r.is_error:
        msg = "The Qwen provider rejected the request."
        if r.status_code in (401,403): msg = "The configured provider token is not authorized for Qwen."
        if r.status_code == 429: msg = "The Qwen provider quota or rate limit was reached."
        raise HTTPException(r.status_code if r.status_code < 500 else 502, msg)
    return Response(r.content, status_code=r.status_code, media_type="application/json")

@app.post("/v1/images/generations")
async def image_generation(payload: ImageRequest, _=Depends(api_key)):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as client:
            r = await client.post(FLUX_URL, headers={**provider_headers(),"Content-Type":"application/json","Accept":"image/*"},
                                  json={"inputs":payload.prompt})
    except httpx.RequestError:
        raise HTTPException(502, "Could not reach the FLUX provider.")
    if r.is_error:
        msg = "The FLUX provider rejected the request."
        if r.status_code in (401,403): msg = "The provider token is not authorized for FLUX.1 Schnell."
        elif r.status_code == 429: msg = "The FLUX provider quota or rate limit was reached."
        elif r.status_code == 503: msg = "The FLUX model is loading or temporarily unavailable; retry shortly."
        raise HTTPException(r.status_code if r.status_code < 500 else 502, msg)
    if not r.headers.get("content-type","").startswith("image/"):
        raise HTTPException(502, "The FLUX provider returned an unexpected response.")
    return {"created":int(datetime.now(timezone.utc).timestamp()),
            "data":[{"b64_json":base64.b64encode(r.content).decode("ascii")}]}

@app.exception_handler(HTTPException)
async def api_errors(request: Request, exc: HTTPException):
    if request.url.path.startswith("/v1/"):
        return JSONResponse(status_code=exc.status_code,
            content={"error":{"message":str(exc.detail),"type":"api_error","code":exc.status_code}},
            headers=exc.headers)
    return JSONResponse(status_code=exc.status_code, content={"detail":exc.detail}, headers=exc.headers)
