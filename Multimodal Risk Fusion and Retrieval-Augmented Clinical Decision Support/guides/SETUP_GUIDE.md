# Step-by-step guide — building and deploying your component

**Component 4 · R26-DS-012** · written to be followed from zero

You will finish with three working things:

1. A trained model on your laptop that turns 5 demographic answers into an anxiety risk score
2. That model running live on the internet as a Hugging Face Space, callable from your app
3. A fusion program that combines your score with your teammates' scores into one final number

**Total time:** about 2–3 hours the first time. Most of it is waiting for downloads.

Work through the parts in order. Don't skip ahead — Part 6 needs Part 5 to have worked.

---

# Part 1 · Get your computer ready

### 1.1 — Install Python

Go to **https://www.python.org/downloads/** and download **Python 3.11** or **3.12**.

> ⚠️ **Do not install Python 3.13.** Some of the libraries we need are not ready for it yet, and you'll get confusing errors an hour from now.

**On Windows**, during installation there is a checkbox at the bottom of the first screen that says **"Add python.exe to PATH"**. **Tick it.** If you miss it, nothing in this guide will work and you'll have to reinstall.

**On Mac**, just run the installer normally.

### 1.2 — Install VS Code

Download from **https://code.visualstudio.com/**. Install it normally.

Then open VS Code and install two extensions. Click the squares icon on the left sidebar (or press `Ctrl+Shift+X`), then search for and install:

- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

### 1.3 — Check it worked

Open a terminal. This is the black text window where you type commands.

- **Windows:** press the Windows key, type `powershell`, press Enter
- **Mac:** press `Cmd+Space`, type `terminal`, press Enter

Type this and press Enter:

```bash
python --version
```

You should see something like `Python 3.12.4`.

> **If you see "command not found" or "not recognised":** on Windows try `py --version` instead. If that works, use `py` everywhere this guide says `python`. If neither works, Python didn't install correctly — reinstall and make sure you tick "Add python.exe to PATH".

---

# Part 2 · Make your project folder

### 2.1 — Create the folders

We're going to keep everything tidy from the start, because in three weeks you will not remember where you put things.

Pick a place you can find easily — your Desktop or Documents folder. In the terminal:

```bash
cd Desktop
mkdir component4
cd component4
mkdir data notebooks fusion hf_space
```

**What these commands mean:**

| Command | What it does |
|---|---|
| `cd Desktop` | "change directory" — go into your Desktop folder |
| `mkdir component4` | "make directory" — create a new folder called `component4` |
| `cd component4` | go inside the folder you just made |
| `mkdir data notebooks ...` | make four folders at once |

### 2.2 — Put the files I gave you in the right places

Download the files from our chat and move them so your folder looks **exactly** like this:

```
component4/
├── data/                          (empty for now — Part 4 fills it)
├── notebooks/
│   └── 01_dcar_demographic_gad7.ipynb
├── fusion/
│   ├── fusion.py
│   ├── demo_fusion.py
│   └── FUSION_DESIGN.md
└── hf_space/
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

> ⚠️ **The exact folder names matter.** The code looks for `data/demographic.csv`. If you name the folder `Data` or `datasets`, it won't find it.

### 2.3 — Open the folder in VS Code

In VS Code: **File → Open Folder** → pick your `component4` folder.

You should now see your folders in the left sidebar.

---

# Part 3 · Create a virtual environment

### 3.1 — What is this and why bother?

A **virtual environment** is a private box of Python libraries that belongs to this project only.

Think of it like a separate pencil case for each subject. Without it, every project on your computer shares one big pile of libraries, and installing something for this project can silently break a project you finished last month. Every professional Python project uses one. It takes 30 seconds.

### 3.2 — Make it

In the terminal, make sure you're inside `component4` (the terminal should show `component4` in the prompt), then:

```bash
python -m venv .venv
```

This creates a hidden folder called `.venv`. Nothing visible happens — that's normal.

### 3.3 — Turn it on ("activate" it)

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

**How you know it worked:** your terminal prompt now starts with `(.venv)`. Like this:

```
(.venv) PS C:\Users\You\Desktop\component4>
```

> **Windows error: "running scripts is disabled on this system"?**
> Windows blocks scripts by default. Run this once, then try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Type `Y` and press Enter when it asks.

> 🔁 **You must activate every time you open a new terminal.** If you close VS Code and come back tomorrow, run the activate command again. If you forget, you'll get "ModuleNotFoundError" and think something is broken when it isn't.

### 3.4 — Install the libraries

With `(.venv)` showing in your prompt:

```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter ipykernel fastapi "uvicorn[standard]" pydantic httpx
```

This downloads about 100 MB and takes 2–5 minutes. You'll see a lot of scrolling text. That's fine.

**What each one is for:**

| Library | Why we need it |
|---|---|
| `pandas` | reads CSV files and holds them as tables |
| `numpy` | fast maths on numbers |
| `scikit-learn` | the machine learning models |
| `matplotlib` | draws the graphs |
| `joblib` | saves your trained model to a file |
| `jupyter`, `ipykernel` | lets you run the notebook |
| `fastapi`, `uvicorn`, `pydantic` | the web service that serves your model |
| `httpx` | lets us test the web service |

### 3.5 — Check nothing is missing

```bash
python -c "import pandas, sklearn, matplotlib, fastapi; print('all good')"
```

If it prints `all good`, you're ready.

---

# Part 4 · Download the dataset

### 4.1 — Get the two files

Go to **https://zenodo.org/records/10423537**

Scroll to the **Files** section. Download **only these two**:

- `demographic.csv` (1.8 MB)
- `gad7.csv` (1.4 MB)

Ignore `isi.csv`, `phq9.csv` and `pss.csv` — those are insomnia, depression and stress scales. We only want anxiety.

### 4.2 — Put them in the data folder

Move both files into `component4/data/`. Your folder should now be:

```
data/
├── demographic.csv
└── gad7.csv
```

### 4.3 — Check they're really there

```bash
python -c "import pandas as pd; d=pd.read_csv('data/demographic.csv'); print(d.shape); print(d.head(3))"
```

You should see roughly `(24292, 6)` and the first three rows with columns `export_id, gender, age, edu, smoke, drink`.

> **"FileNotFoundError"?** You're either in the wrong folder (type `cd` and check), or the files went into `Downloads` instead of `data`. Check with `ls data` (Mac) or `dir data` (Windows).

### 4.4 — Know what this data is

24,292 **Chinese university students**, surveyed February–March 2021, during COVID. Your patients are **Sri Lankan psychiatric inpatients**.

That is a big difference and you must write it in your limitations section. The model learns the *shape* of the relationship — which kinds of profiles carry more risk — not the exact probabilities for a Sri Lankan ward. You will re-calibrate on real NHSL data later.

---

# Part 5 · Run the notebook

### 5.1 — Open it

In VS Code, click `notebooks` in the sidebar, then click `01_dcar_demographic_gad7.ipynb`.

### 5.2 — Pick the right Python

Top-right of the notebook there's a button that says **"Select Kernel"**. Click it → **Python Environments** → pick the one with `.venv` in the name.

> This tells the notebook to use your project's pencil case, not some other Python on your computer. If you pick the wrong one you'll get import errors.

### 5.3 — ⚠️ THE ONE LINE YOU MUST CHANGE

Scroll to the cell titled **"1 · Configuration"**. Near the bottom find:

```python
SYNTHETIC_FALLBACK = True            # <- set False once the real CSVs are in ./data
```

**Change `True` to `False`:**

```python
SYNTHETIC_FALLBACK = False
```

**Why this matters more than anything else in this guide:** when this is `True` and the CSVs are missing, the notebook invents fake data so the code can be tested. It runs fine and prints beautiful numbers. Those numbers are **completely meaningless**. If you leave it on `True` by accident and put the results in your paper, you will have published made-up findings. Set it to `False` and the notebook will refuse to run without real data, which is exactly what you want.

### 5.4 — Run everything

Click **"Run All"** at the top. Say yes if it asks to install anything.

Takes about 3–8 minutes. The slowest part is the permutation test (it trains the model 200 times on shuffled data).

> **Want a fast first pass?** Temporarily set `N_PERMUTATIONS = 20` to check everything works, then set it back to `200` for the run you actually report. Never report results from a 20-permutation run.

---

# Part 6 · Read your results

Don't just check that it finished. Go through these five checkpoints in order.

### ✅ Checkpoint 1 — Did it load the real data?

Near the top you should **NOT** see the block of `!!!!!!` warning lines. If you see them, you didn't set `SYNTHETIC_FALLBACK = False`. Go back to 5.3.

You should see roughly `merged cohort : 24,292 participants`.

### ✅ Checkpoint 2 — Any UNMAPPED categories?

Search the output (`Ctrl+F`) for the word `UNMAPPED`.

If you see lines like:

```
!! UNMAPPED in `edu` -> ['junior college', 'vocational']
```

**Stop and fix this.** It means the notebook found education levels it doesn't recognise, and it quietly replaced them with the middle value. That destroys real signal.

**How to fix:** go to the cell with `EDU_ORDER` and add the missing words to whichever line they belong to:

```python
EDU_ORDER = [
    (["no formal", "primary", "less than high"],                       0),
    (["high school", "secondary", "o/l", "ordinary level"],             1),
    (["some college", "diploma", "a/l", "advanced level", "associate",
      "junior college", "vocational"],                                  2),   # <- added
    (["bachelor", "undergrad", "b.sc", "bsc"],                          3),
    ...
]
```

Then Run All again. Repeat until no `UNMAPPED` lines appear. Do the same for `smoke` and `drink`.

### ✅ Checkpoint 3 — Did the model beat random chance?

Find the permutation test output:

```
observed 0.6612  ->  p = 0.0050   CLEARS the null
```

- **"CLEARS the null"** → your model found real signal. Good.
- **"DOES NOT clear the null"** → your model is no better than guessing. This is a genuine finding, not a bug. Tell your supervisor, and set `CLEARS_PERMUTATION_NULL["c4_demographic"] = False` in `fusion.py` — your own rule then excludes your own component, which is uncomfortable but honest and academically strong.

### ✅ Checkpoint 4 — What are your actual numbers?

Find the `── TEST SET ──` block:

```
AUROC   0.6612   95% CI [0.6480, 0.6741]
AUPRC   0.4103   95% CI [...]
Brier   0.1832
ECE     0.0221
```

**Write the AUROC number down. You need it in Part 8.**

**How to read AUROC:** 0.5 = coin flip, 1.0 = perfect. Anything **0.60–0.70 is expected and fine here.** Demographics are a weak predictor of anxiety — that's a known fact about the world, not a failure of your model. What makes it publishable is that you measured it honestly against a null.

**ECE** is calibration error. Under 0.05 is good. This matters because your score gets multiplied by a weight in the fusion — a badly calibrated probability poisons the composite no matter how well you tune the weights.

### ✅ Checkpoint 5 — Are the files saved?

At the very bottom:

```
round-trip OK: reloaded score 0.0313
artefacts: ['dcar_metadata.json', 'dcar_model.joblib', 'dcar_reference_scores.npy', 'fig_...png']
```

`round-trip OK` means the saved model gives the same answer as the notebook. If that assertion fails, don't deploy — something is wrong with the save.

You now have a folder `notebooks/artefacts/` containing your trained model and four graphs for your paper.

---

# Part 7 · Test the web service on your own computer

Before putting it on the internet, check it works locally.

### 7.1 — Copy the model into the service folder

From inside `component4`:

**Windows:**
```bash
mkdir hf_space\artefacts
copy notebooks\artefacts\dcar_model.joblib hf_space\artefacts\
```

**Mac / Linux:**
```bash
mkdir -p hf_space/artefacts
cp notebooks/artefacts/dcar_model.joblib hf_space/artefacts/
```

### 7.2 — Start the service

```bash
cd hf_space
uvicorn app:app --reload --port 7860
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:7860
```

Leave this terminal running. It's your server now.

### 7.3 — Try it in your browser

Open **http://127.0.0.1:7860/docs**

You'll see an automatic testing page. Click **POST /predict** → **"Try it out"** → paste this into the box:

```json
{
  "patient_id": "TEST-001",
  "gender": "female",
  "age": 21,
  "edu": "bachelor's degree",
  "smoke": "never smokes",
  "drink": "never drinks"
}
```

Click **Execute**. Scroll down and you should get something like:

```json
{
  "patient_id": "TEST-001",
  "score": 0.3421,
  "risk_label": "elevated",
  "severity_probs": {"Minimal": 0.41, "Mild": 0.25, "Moderate": 0.20, "Severe": 0.14},
  "expected_gad7": 7.2,
  "confidence": 0.31,
  "coverage": 1.0,
  "available": true
}
```

**🎉 That is your model working.** `score` is what goes into the fusion.

### 7.4 — Try a broken input on purpose

Now send one with fields missing:

```json
{ "patient_id": "TEST-002", "gender": "male", "age": 34 }
```

`coverage` should drop to `0.4` (2 of 5 fields) and `available` should be `false`. That's the service correctly telling the fusion layer "don't trust me, I barely know anything about this person."

### 7.5 — Stop the server

Press `Ctrl+C` in that terminal.

---

# Part 8 · Run the fusion

### 8.1 — Put your real AUROC into the fusion

Open `fusion/fusion.py`. Find:

```python
VALIDATION_AUROC = {
    "c1_physiological": 0.6191,
    "c2_behavioral": 0.5205,
    "c3_clinical_nlp": 0.7380,
    "c4_demographic": 0.6600,     # <-- REPLACE with the test AUROC from your notebook
}
```

Replace `0.6600` with the AUROC you wrote down in Checkpoint 4.

**Why:** the weight your component gets in the final score is calculated from how much it beats chance. Leaving a guessed number there means your fusion weights are based on a number you made up — and that is exactly the kind of thing an examiner asks about.

### 8.2 — Run the demo

```bash
cd ../fusion
python demo_fusion.py
```

### 8.3 — What you're looking at

The output shows the same three risk scores fused five different ways. **Only the age of each reading changes.** Watch the weights:

| Situation | physio weight | notes weight | your weight |
|---|---|---|---|
| Everything fresh | 0.216 | 0.484 | 0.300 |
| Strap off 3 hours | **0.004** | 0.646 | 0.350 |
| Note is 3 months old | **0.503** | **0.147** | 0.350 |

Row 2: the chest strap reading is 3 hours old, so it almost vanishes — correct, because your heart rate this morning says nothing about now.

Row 3: the clinical note is 3 months old, so live physiology takes over instead.

**Nobody typed those numbers in.** They come out of the half-life formula. That's the whole argument for the design.

Then look at **Scenario D**: on day one, when you only have demographics, the system returns **no tier at all** — not "Low risk". Saying "low risk" about a newly admitted psychiatric patient based only on their age and education would be dangerous. Be ready to explain this one; it's the safety decision an examiner is most likely to probe.

And **Scenario E**: the behavioural stream sends 0.95 (very high) and the composite doesn't move at all, because that component failed its permutation null and carries zero weight.

---

# Part 9 · Put it on Hugging Face

### 9.1 — Make an account

Sign up at **https://huggingface.co/join**. Free.

### 9.2 — Create a Space

Go to **https://huggingface.co/new-space**:

- **Space name:** `dcar-demographic-risk`
- **License:** MIT
- **Space SDK:** **Docker** → **Blank**
- **Visibility:** **Private** (you can make it public later; keep patient-adjacent work private by default)

Click **Create Space**.

### 9.3 — Install the tools

Back in your terminal, with `(.venv)` active:

```bash
pip install huggingface_hub
```

You also need **git** — check with `git --version`. If missing, get it from **https://git-scm.com/downloads**.

And **Git LFS**, which handles files too big for normal git:

```bash
git lfs install
```

If that fails, install from **https://git-lfs.com** first.

### 9.4 — Log in

```bash
hf auth login
```

It asks for a token. Get one at **https://huggingface.co/settings/tokens** → **Create new token** → give it **Write** permission → copy it → paste into the terminal.

> The terminal won't show anything as you paste the token. That's a security feature, not a bug. Just paste and press Enter.

> If `hf auth login` isn't recognised, your version is older — use `huggingface-cli login` instead.

### 9.5 — Upload

From inside `component4`:

```bash
cd hf_space
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/dcar-demographic-risk
git lfs track "*.joblib"
git add .gitattributes
git add .
git commit -m "DCAR v1.0 — demographic anxiety risk model"
git push --force origin main
```

Replace `YOUR_USERNAME` with your actual Hugging Face username.

> `git lfs track "*.joblib"` **must** come before `git add .` — otherwise the model file gets committed the wrong way and the push may be rejected.

### 9.6 — Watch it build

Go to `https://huggingface.co/spaces/YOUR_USERNAME/dcar-demographic-risk`.

You'll see **"Building"** for 2–4 minutes, then **"Running"**.

If it says **"Build error"**, click the **Logs** tab and read the last few red lines. Most common cause: `artefacts/dcar_model.joblib` wasn't uploaded. Check it exists in the **Files** tab of the Space.

### 9.7 — Add the password

In your Space: **Settings** → **Variables and secrets** → **New secret**

- Name: `DCAR_API_TOKEN`
- Value: any long random string you invent, e.g. `r26ds012-dcar-8f3k2n9x`

Save it somewhere. Your apps need it.

> Without this secret, anyone who finds your URL can use your model. With patient-adjacent work, always set it.

### 9.8 — Test the live service

```bash
curl -X POST https://YOUR_USERNAME-dcar-demographic-risk.hf.space/predict \
  -H "Authorization: Bearer r26ds012-dcar-8f3k2n9x" \
  -H "Content-Type: application/json" \
  -d "{\"patient_id\":\"TEST-001\",\"gender\":\"female\",\"age\":21,\"edu\":\"bachelor's degree\",\"smoke\":\"never smokes\",\"drink\":\"never drinks\"}"
```

Same JSON as Part 7, but now coming from the internet. **Your model is live.**

> **First call takes ~20 seconds** because the Space was asleep. Later calls are fast. This doesn't matter for you — your model runs once per patient at registration, so it's never on a time-critical path.

---

# Part 10 · When things go wrong

| What you see | What it means | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Virtual environment isn't active | Run the activate command from 3.3. Check for `(.venv)` in your prompt |
| `FileNotFoundError: data/demographic.csv` | Wrong folder, or CSVs elsewhere | `cd` to `component4`, check the files are in `data/` |
| Notebook prints `!!!!!!` warnings | `SYNTHETIC_FALLBACK` still `True` | Step 5.3 |
| `UNMAPPED in edu -> [...]` | Categories the code doesn't recognise | Checkpoint 2 — add the words to `EDU_ORDER` |
| `DOES NOT clear the null` | Model is no better than chance | Not a bug. Real finding. See Checkpoint 3 |
| `serving path diverges from notebook!` | Saved model ≠ notebook model | Restart kernel, Run All again. Don't deploy until it passes |
| Space stuck on "Build error" | Usually a missing model file | Space → Files tab → check `artefacts/dcar_model.joblib` is there |
| `git push` rejected | LFS not set up before adding | Delete `.git` folder, redo 9.5 in exact order |
| PowerShell "scripts is disabled" | Windows security default | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Everything worked yesterday, broken today | New terminal, venv not active | Activate again (3.3). This will happen to you many times |

---

# Part 11 · What to do next

**This week:**

1. Run everything above, get your real AUROC, and paste the `── TEST SET ──` block into a document — that's your results section starting.
2. Save the four PNGs from `notebooks/artefacts/` — those are your paper's figures.
3. Message your three teammates and ask each of them to add four fields to their existing response: `confidence`, `coverage`, `captured_at`, `model_version`. Show them `FUSION_DESIGN.md` §5 so they can see why. **Do this now** — retrofitting it in week 10 is painful and they will say no.
4. Agree with the patient-app and clinician-app owners on one question: **how does `user_id` in the patient app map to `mrn` in the doctor's app?** Every join in your component depends on this and it is the most common place projects like yours break.

**Not yet built** (say so plainly if asked in a review — a known gap is fine, a hidden one is not):

- The `/fuse` web service that wraps `fusion.py` and calls your teammates' Spaces
- The RAG layer for the doctor's app
- Connecting the patient app's demographics to your Space instead of Google Sheets

**One thing to be careful about.** Your model was trained on Chinese university students and will be used on Sri Lankan psychiatric patients. It transfers as a *pattern*, not as exact probabilities. Once you have 40+ real NHSL patients with GAD-7 scores, re-fit the isotonic calibration step on those. Until then, present the score as a **relative** ranking within your cohort, not as "this patient has a 34% chance of anxiety."
