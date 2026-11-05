# DAT09-06 — Representation Learning

**M2 ADVANCED · 18 hours · 6 sessions × 3h**

This is the working repository for the course: the environment you install, the
shared utilities the notebooks import, and the data build step.

Everything runs **on a CPU laptop**. There is no GPU anywhere in this course, by
design: no in-class training run should take more than five minutes.

---

## Set up before Monday 9 November

Session 2 starts with code from the first minute, so do this at home. It takes
about ten minutes, most of it waiting for `pip`.

### 1 · Get the repository

```bash
git clone https://github.com/PierreFontaineParalos/DAT09-06-representation-learning-students.git
cd DAT09-06-representation-learning-students
```

### 2 · Create and activate a virtual environment

```bash
python -m venv .venv
```

Then activate it:

| Your shell | Command |
|---|---|
| macOS, Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

You should see `(.venv)` at the start of your prompt. If you do not, the
environment is not active and the next step will install into the wrong place.

### 3 · Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A few minutes: PyTorch alone is several hundred MB.

**Linux users only** — the default wheels drag in ~2.5 GB of CUDA libraries you
will never use. Install from the CPU index instead:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4 · Build the data

```bash
python data/prepare_data.py --all
```

Downloads Fashion-MNIST and rebuilds it identically on every machine. About
30 MB, five seconds.

### 5 · Check your environment

```bash
jupyter lab
```

Open **`environment-check.ipynb`** and run it top to bottom (*Run → Run All
Cells*). It checks your Python version, every library, PyTorch on the CPU, a
real training loop, the course utilities, figures and the data.

The last cell prints a **verdict**. That is what I want to see on Monday.

---

## Requirements

- Python **3.10 – 3.14**
- ~3 GB of free disk space
- No GPU

---

## If you get stuck

| Symptom | Likely cause |
|---|---|
| `python` not found, or version < 3.10 | Try `python3`; otherwise install Python from python.org |
| No `(.venv)` in front of the prompt | The environment is not activated: redo step 2 |
| The course utilities cell fails | The notebook was opened outside the repository folder |
| A `parquet` error | Redo step 3: `pyarrow` is in `requirements.txt` |

Before you write to me:

1. Read the error message **in full**.
2. Ask an AI, **explaining in your own words** what you did and what you
   expected — not just pasting the error.

Still stuck? Email me **before Sunday evening**, with the output of the
notebook: `fontaine.pie@protonmail.com`. On Monday there is no time to debug an
installation.

---

## Using AI in this course

You can code with AI — ChatGPT, Claude, Copilot, Cursor, Gemini — without
restriction, in the labs and at home. It is a tool of your trade.

But you must be able to **explain every line you hand in**. In the lab I may
point at a line and ask what it does, why it is there, and what breaks if we
remove it. "The AI wrote it" is not an answer.

The final exam is a **multiple-choice test with no outside help**. It does not
ask you to be a Python expert; it asks whether you understood the concepts.

---

## What is in here

```
requirements.txt           the pinned CPU-only environment
environment-check.ipynb    run this before session 2
common/                    utilities the notebooks import
data/prepare_data.py       builds the datasets
```

Session folders and notebooks are added as the course goes.
