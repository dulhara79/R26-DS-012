# Connecting to Hugging Face — step by step

**Component 4 · R26-DS-012** · written to be followed with no prior knowledge

You have two separate jobs here. Don't mix them up:

- **Job A** — *host your own model* on Hugging Face, so the others can reach your score
- **Job B** — *connect to your teammates' models*, which are already on Hugging Face, so your fusion can pull their scores

We'll do **Job A first** (you control it, so it's predictable), then **Job B**.

Set aside about 90 minutes. Have your DCAR notebook finished, because Job A needs the model file it produced.

---

# Background: what a "Space" actually is (2 minutes)

A **Hugging Face Space** is just a small computer on the internet that runs your code and waits for questions. You send it a patient's details, it sends back a risk score. That's it.

Two things that will confuse you if nobody warns you:

1. **Spaces fall asleep.** If nobody uses a free Space for a while, it goes to sleep to save power. The *first* request after that takes 20–40 seconds while it wakes up. Every request after is fast. This is normal — not a bug, not your fault.

2. **Every Space is different inside.** Your teammates each built their own. One might answer at `/predict`, another at `/score`. One might call the score `score`, another `risk_score`. There is no rule they all follow. That's why Job B starts with a tool that *looks* at each Space to find out.

---

# JOB A — Put your own model on Hugging Face

## A1 · Make a Hugging Face account

Go to **https://huggingface.co/join** and sign up. Free. Confirm your email.

## A2 · Get the files ready

You need the folder I gave you called `hf_space`, and your trained model file. Your folder should look like this:

```
hf_space/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
└── artefacts/
    └── dcar_model.joblib      ← from your notebook
```

If `artefacts/dcar_model.joblib` isn't there yet, copy it from your notebook's output:

**Mac/Linux:**
```bash
mkdir -p hf_space/artefacts
cp notebooks/artefacts/dcar_model.joblib hf_space/artefacts/
```

**Windows:**
```powershell
mkdir hf_space\artefacts
copy notebooks\artefacts\dcar_model.joblib hf_space\artefacts\
```

> ⚠️ Remember from before: that model file is currently trained on the Chinese student data (or synthetic, if you haven't switched `SYNTHETIC_FALLBACK` off). That's fine for getting the plumbing working. You'll re-upload a better model later — uploading is a 30-second repeat.

## A3 · Create the Space on the website

Go to **https://huggingface.co/new-space**. Fill in:

- **Owner:** your username
- **Space name:** `dcar-demographic-risk`
- **License:** MIT
- **Select the Space SDK:** click **Docker**, then **Blank**
- **Space hardware:** CPU basic (free) is fine
- **Visibility:** **Private** for now

Click **Create Space**. You now have an empty Space with a page like `https://huggingface.co/spaces/YOURNAME/dcar-demographic-risk`.

## A4 · Install the upload tools (one time)

Back in your terminal, with your virtual environment active (`(.venv)` showing):

```bash
pip install huggingface_hub
```

You also need **git** and **git-lfs**. Check:

```bash
git --version
git lfs version
```

If `git` is missing → install from **https://git-scm.com/downloads**.
If `git lfs` is missing → install from **https://git-lfs.com**, then run `git lfs install` once.

> **What is git-lfs and why?** Normal git is built for text files. Your model file is a chunk of binary data. Git LFS ("Large File Storage") is the piece that handles big non-text files properly. Without it, the upload of your `.joblib` can get corrupted.

## A5 · Log in to Hugging Face from the terminal

```bash
hf auth login
```

It asks for a **token**. Get one here: **https://huggingface.co/settings/tokens** → **Create new token** → choose type **Write** → **Create** → copy it.

Paste it into the terminal and press Enter.

> When you paste the token, **nothing appears on screen**. No dots, no stars. That's a security feature, not a freeze. Just paste and hit Enter.

> If `hf auth login` says "command not found", your version is older — use `huggingface-cli login` instead. Same thing.

## A6 · Upload

Go into your `hf_space` folder and run these lines **one at a time**. Replace `YOURNAME` with your real username.

```bash
cd hf_space
git init
git remote add origin https://huggingface.co/spaces/YOURNAME/dcar-demographic-risk
git lfs track "*.joblib"
git add .gitattributes
git add .
git commit -m "DCAR v1.0 demographic anxiety risk model"
git branch -M main
git push --force origin main
```

> ⚠️ The order matters. `git lfs track "*.joblib"` **must** come before `git add .`. If you add the model file before telling LFS to handle it, the upload may be rejected and you'll have to start A6 over (delete the hidden `.git` folder first).

> If `git push` asks for a username and password: username is your HF username, and the "password" is the **token** from A5 (not your account password).

## A7 · Watch it build

Open your Space page in a browser: `https://huggingface.co/spaces/YOURNAME/dcar-demographic-risk`

You'll see **Building** for 2–4 minutes, then **Running**.

If it says **Build error** or **Runtime error**, click the **Logs** tab and read the last few red lines. The most common cause is the model file didn't upload — check the **Files** tab for `artefacts/dcar_model.joblib`.

## A8 · Add a password to your Space

Anyone who finds the URL could use your model. Lock it:

In your Space → **Settings** → **Variables and secrets** → **New secret**:
- Name: `DCAR_API_TOKEN`
- Value: invent a long string, e.g. `r26ds012-dcar-8f3k2n9x`

Save it. Write it down — your fusion service needs it in Job B.

## A9 · Prove it's alive

In your terminal:

```bash
curl https://YOURNAME-dcar-demographic-risk.hf.space/health
```

> Note the URL shape: on the website it's `huggingface.co/spaces/YOURNAME/...`, but the *live app* is at `YOURNAME-spacename.hf.space`. Dashes, not slashes. This trips everyone up once.

You should get back JSON with your model version. **Your model is now live on the internet.** That's Job A done.

---

# JOB B — Connect to your teammates' Spaces

Your fusion service needs three things from each teammate's Space:

1. its **URL**
2. its **token** (if private)
3. what its response **looks like** — the endpoint path and the score's field name

You get #1 and #2 by asking them. You get #3 with the probe tool.

## B1 · Ask each teammate for two things

Send this message to each of C1, C2 (if used), and C3:

> "For the fusion, I need:
> 1. The live URL of your Space (the `something.hf.space` one, not the huggingface.co one)
> 2. The access token, if your Space is private
> 3. And separately — a CSV of the risk scores your model produced on your **held-out test set** (just the numbers). I need it to line up our score scales."

The CSV is Ask 2 from before — you'll use it in Part 7 of the fusion guide. Get it now while they're still working.

## B2 · Look inside each Space with the probe tool

You don't need to understand their code. `probe_space.py` pokes their Space and tells you exactly how to talk to it.

With your fusion service's virtual environment active, in the `fusion_service` folder:

```bash
python probe_space.py https://THEIR-space-url.hf.space
```

If their Space is private, add their token:

```bash
python probe_space.py https://THEIR-space-url.hf.space --token hf_theirtoken
```

**First run will be slow** — it's waking their Space up. Wait for it.

## B3 · Read the probe's answer

At the bottom, under **WHAT TO DO NEXT**, it tells you three things:

```
It works. Use this:

  URL     : https://their-space.hf.space/predict
  Method  : POST
  Send    : {"patient_id": "TEST-001", "mrn": "TEST-001"}

It returned:
  { "risk_score": 0.8213, "confidence": 0.779, ... }

The score is here:  response.risk_score  (field name: 'risk_score', value 0.8213)
```

The two things you're hunting for:

- **the URL** — copy it exactly
- **the field name of the score** — here it's `risk_score`

If the probe warns you the field name is unusual (like `risk_score` instead of `score`), it tells you to add it to your `clients.py`. Do that: open `clients.py`, find the line inside `to_reading()`:

```python
for key in ("score", "risk_score", "value", "probability", "risk"):
```

and add their field name to the list if it's not already there. `risk_score` is already covered; something oddball like `"anxiety_output"` you'd add yourself.

## B4 · If the probe finds nothing

Some Spaces (especially Gradio ones) don't answer to simple pokes. Then:

1. Open `https://THEIR-space.hf.space/docs` in a browser — if it's a FastAPI Space, this page lists every endpoint and you can read the shape there.
2. If that fails too, just ask the teammate directly: *"What's the exact URL and the exact JSON I send to get a risk score back? Paste me a working example."* One message saves an hour.

## B5 · Put all the URLs in your `.env` file

In your `fusion_service` folder, make a copy of `.env.example` and call it `.env`:

**Mac/Linux:** `cp .env.example .env`
**Windows:** `copy .env.example .env`

Open `.env` and fill in the real values you gathered:

```bash
C1_URL=https://teammate1-physio.hf.space/predict
C2_URL=
C3_URL=https://teammate3-tcwpn.hf.space/predict
C4_URL=https://YOURNAME-dcar-demographic-risk.hf.space/fusion_component

C1_TOKEN=hf_teammate1token
C2_TOKEN=
C3_TOKEN=hf_teammate3token
C4_TOKEN=r26ds012-dcar-8f3k2n9x

COMPONENT_TIMEOUT_S=8
FUSION_API_TOKEN=r26ds012-fusion-picksomething
```

Notes:
- **C2 is left blank on purpose** — behavioural gets zero weight, so we don't call it.
- **C4 is your own Space**, using the token you set in A8.
- Leave a token blank if that Space is public.

## B6 · Make the fusion service read the `.env` file

Open `app.py` and add these two lines at the very top (right after the docstring, before the other imports):

```python
from dotenv import load_dotenv
load_dotenv()
```

Then install the reader (once):

```bash
pip install python-dotenv
```

## B7 · Protect your secrets

Your `.env` now contains real tokens. **Never** put it on GitHub. In the `fusion_service` folder, make a file called `.gitignore` containing one line:

```
.env
```

## B8 · Run it against the real Spaces

```bash
uvicorn app:app --reload --port 7861
```

Open **http://127.0.0.1:7861/docs**, find **POST /v1/fuse**, click **Try it out**, send:

```json
{ "mrn": "NHSL-0142" }
```

Click **Execute**. The first call may take 30+ seconds (waking sleeping Spaces). You should get a composite score and a tier, with a `harmonisation` block showing each teammate's raw score and its percentile.

**If a component shows `unavailable`:** read its `note`. `timeout` = their Space was asleep, try again. `HTTP 404` = wrong URL path, re-check with the probe. `no numeric score field` = their field name isn't in your `clients.py` list (B3).

---

# The order to actually do this in

1. **Job A** — get your own Space live and tested (A1–A9). Do this first; you control it.
2. **Message teammates** — URLs, tokens, and the held-out CSVs (B1). Do this today; it unblocks everything.
3. **Probe each Space** as their URLs arrive (B2–B4). You don't have to wait for all three at once.
4. **Fill `.env`, wire it up, test** (B5–B8).
5. Then Part 7 of the fusion guide — build the real reference distributions from those CSVs.

---

# Quick reference — the two URL shapes

This confuses everyone, so keep it handy:

| You want to... | Use this shape |
|---|---|
| Open the Space's **page** (settings, logs, files) | `https://huggingface.co/spaces/NAME/SPACE` |
| Have your code **call** the Space | `https://NAME-SPACE.hf.space/endpoint` |

Slashes for the page. Dashes for the live app.

---

# If you get stuck

| Problem | Likely cause | Fix |
|---|---|---|
| `curl` to your Space returns nothing for 30s then works | Space was asleep | Normal. It's awake now |
| Space page shows "Build error" | Model file missing, or a typo in a file | Logs tab → read the last red lines |
| `git push` rejected | LFS not set before `git add` | Delete `.git` folder, redo A6 in order |
| push asks for password | It wants your token, not your password | Paste the Write token from A5 |
| Probe says "could not reach" | Wrong URL, or private without token | Check the `.hf.space` URL; add `--token` |
| `/v1/fuse` shows a component `unavailable` | See the `note` field | timeout→retry, 404→wrong path, field→edit clients.py |
| Everything worked, now `ModuleNotFoundError` | New terminal, venv off | Reactivate the venv |
