import os
import re
import sys
import glob
import math
import shutil
import subprocess
import threading
import collections
import time as time_module

import numpy as np
import sounddevice as sd
import requests

# --- Configuration ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

PEAK_THRESHOLD = 0.35
TRANSIENT_RATIO = 4.0
COOLDOWN = 2.0

REQUEST_TIMEOUT = 300  # seconds; generous so a slow first response never times out mid-demo

WATCH_FOLDER = "."  # <-- change to the folder you'll be coding in during the demo

# If set to a filename (e.g. "main.py"), THAT exact file is always the target
# for both reading (data sent to the AI) and writing (the correction) - no
# folder scanning happens at all. Leave as None to auto-detect the most
# recently saved code file in WATCH_FOLDER instead.
MAIN_FILE = None

# Editor used to open the fixed file. This uses the editor's own CLI command
# so the RIGHT app opens even if the OS has a different app set as default
# for this file type (e.g. Antigravity) - VS Code = "code", Cursor = "cursor",
# Windsurf = "windsurf". Set to None to skip this and fall back to the OS
# default-app behaviour below.
EDITOR_CLI_COMMAND = "code"

# Fallback only: used if EDITOR_CLI_COMMAND is None or not found on PATH.
# Mac examples: "Visual Studio Code", "Cursor", "Antigravity"
# Set to None to skip opening entirely (editor will still auto-reload the file on its own).
EDITOR_APP_NAME = "Visual Studio Code"

# Supported languages: extension -> human-readable name (used in the prompt)
LANGUAGE_MAP = {
    ".py": "Python",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".java": "Java",
}

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_EMBED_URL = "http://127.0.0.1:1234/v1/embeddings"
LM_STUDIO_MODEL = "google/gemma-4-e4b"

# --- RAG configuration ---
# Set RAG_ENABLED = False to turn the whole retrieval pipeline off.
RAG_ENABLED = True
# Embedding model identifier as shown in LM Studio. If this model isn't loaded,
# the pipeline automatically falls back to lexical (keyword) retrieval instead.
RAG_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
RAG_CHUNK_LINES = 40        # how many lines per indexed chunk
RAG_CHUNK_OVERLAP = 10      # overlapping lines between consecutive chunks
RAG_TOP_K = 4               # how many chunks to inject into the prompt
RAG_SCAN_SUBFOLDERS = True  # index the whole project tree, not just one folder
RAG_MAX_FILE_BYTES = 200_000  # skip enormous files

recent_peaks = collections.deque(maxlen=5)
last_trigger = 0
is_fixing = False

# Absolute path of this watcher script itself, so auto-detect never targets
# itself even if it happens to sit inside WATCH_FOLDER.
_SELF_PATH = os.path.abspath(__file__)

# Cached RAG index: rebuilt only when the set of files / their mtimes change.
_rag_index = []          # list of dicts: {file, start_line, text, vector}
_rag_signature = None    # fingerprint of the folder state the index was built from
_rag_mode = "none"       # "embeddings" | "lexical" | "none"
_rag_idf = {}            # token -> inverse document frequency weight
_rag_corpus_size = 0     # number of indexed chunks (for default IDF)


def get_latest_code_file(folder):
    """Return the file to send to the AI / write the correction back to.

    - If MAIN_FILE is set, that exact file is always the target (locked).
    - Otherwise, auto-detect the most recently modified code file in
      `folder`, excluding backups (.bak), this watcher script itself, and
      obvious test files (test_*.* / *_test.*) so a stray helper/test file
      never gets picked up as "the file I just saved".
    """
    if MAIN_FILE:
        path = MAIN_FILE if os.path.isabs(MAIN_FILE) else os.path.join(folder, MAIN_FILE)
        return path if os.path.isfile(path) else None

    def is_candidate(f):
        if not os.path.isfile(f) or f.endswith(".bak"):
            return False
        if os.path.abspath(f) == _SELF_PATH:
            return False
        name = os.path.basename(f).lower()
        stem, _ = os.path.splitext(name)
        if stem.startswith("test_") or stem.endswith("_test"):
            return False
        return True

    all_files = []
    for ext in LANGUAGE_MAP:
        all_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    all_files = [f for f in all_files if is_candidate(f)]
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# RAG pipeline (index -> retrieve -> augment prompt)
# ---------------------------------------------------------------------------

def _iter_project_files(folder):
    """Yield every supported code file in the project (optionally recursive)."""
    for ext in LANGUAGE_MAP:
        pattern = os.path.join(folder, "**", f"*{ext}") if RAG_SCAN_SUBFOLDERS \
            else os.path.join(folder, f"*{ext}")
        for path in glob.glob(pattern, recursive=RAG_SCAN_SUBFOLDERS):
            if not os.path.isfile(path) or path.endswith(".bak"):
                continue
            if os.path.abspath(path) == _SELF_PATH:
                continue
            try:
                if os.path.getsize(path) > RAG_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def _folder_signature(folder):
    """Cheap fingerprint of the project state, so we only re-index when needed."""
    parts = []
    for path in sorted(_iter_project_files(folder)):
        try:
            parts.append(f"{path}:{os.path.getmtime(path)}")
        except OSError:
            continue
    return "|".join(parts)


def _chunk_file(path):
    """Split one file into overlapping line-windows. Returns list of chunk dicts."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()  # splitlines() handles both \r\n and \n correctly
    except Exception:
        return []

    chunks = []
    step = max(1, RAG_CHUNK_LINES - RAG_CHUNK_OVERLAP)
    for start in range(0, max(1, len(lines)), step):
        window = lines[start:start + RAG_CHUNK_LINES]
        if not any(line.strip() for line in window):
            continue
        chunks.append({
            "file": path,
            "start_line": start + 1,
            "text": "\n".join(window),
            "vector": None,
        })
        if start + RAG_CHUNK_LINES >= len(lines):
            break
    return chunks


def _embed_texts(texts):
    """Get embedding vectors from LM Studio. Returns None if unavailable."""
    try:
        response = requests.post(
            LM_STUDIO_EMBED_URL,
            json={"model": RAG_EMBED_MODEL, "input": texts},
            timeout=60,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
    except Exception:
        return None


def _tokenize(text):
    """Very small tokenizer for the lexical fallback: identifiers and words."""
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text)]


def _build_idf(chunks):
    """Inverse document frequency over the chunk corpus.

    Without this, boilerplate tokens ('def', 'return', 'import', 'self') score
    the same as a distinctive function name, and an unrelated file can easily
    outrank the file that actually defines the symbol you called. IDF pushes
    common tokens toward zero weight and rare identifiers toward high weight.
    """
    n = len(chunks)
    doc_freq = collections.Counter()
    for chunk in chunks:
        for token in set(_tokenize(chunk["text"])):
            doc_freq[token] += 1
    return {t: math.log((n + 1) / (d + 0.5)) for t, d in doc_freq.items()}, n


def _lexical_score(query_tokens, chunk_text, idf, corpus_size):
    """TF-IDF style overlap score used when no embedding model is available."""
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    chunk_counts = collections.Counter(chunk_tokens)
    default_idf = math.log(corpus_size + 1)
    score = 0.0
    for token in set(query_tokens):
        if token in chunk_counts:
            # log-damped term frequency, weighted by how rare the token is
            tf = 1.0 + math.log(chunk_counts[token])
            score += tf * idf.get(token, default_idf)
    return score / math.sqrt(len(chunk_tokens))


def build_rag_index(folder):
    """Build (or reuse) the chunk index for the project. Safe to call every snap."""
    global _rag_index, _rag_signature, _rag_mode, _rag_idf, _rag_corpus_size

    if not RAG_ENABLED:
        _rag_index, _rag_mode = [], "none"
        return

    signature = _folder_signature(folder)
    if signature == _rag_signature and _rag_index:
        return  # nothing changed since last build - reuse the cached index

    chunks = []
    for path in _iter_project_files(folder):
        chunks.extend(_chunk_file(path))

    if not chunks:
        _rag_index, _rag_signature, _rag_mode = [], signature, "none"
        return

    vectors = _embed_texts([c["text"] for c in chunks])
    if vectors and len(vectors) == len(chunks):
        for chunk, vector in zip(chunks, vectors):
            norm = np.linalg.norm(vector)
            chunk["vector"] = vector / norm if norm > 0 else vector
        _rag_mode = "embeddings"
    else:
        _rag_mode = "lexical"

    # IDF is always built - the lexical path is also the fallback if an
    # embedding request fails later at query time.
    _rag_idf, _rag_corpus_size = _build_idf(chunks)

    _rag_index = chunks
    _rag_signature = signature
    print(f"[RAG] Indexed {len(chunks)} chunks from "
          f"{len(set(c['file'] for c in chunks))} files (mode: {_rag_mode})")


def retrieve_context(query_text, target_path, top_k=RAG_TOP_K):
    """Return the most relevant chunks from OTHER files as a prompt block."""
    if not RAG_ENABLED or not _rag_index:
        return ""

    target_abs = os.path.abspath(target_path)
    candidates = [c for c in _rag_index if os.path.abspath(c["file"]) != target_abs]
    if not candidates:
        return ""

    scored = []
    if _rag_mode == "embeddings":
        query_vectors = _embed_texts([query_text])
        if query_vectors:
            q = query_vectors[0]
            norm = np.linalg.norm(q)
            q = q / norm if norm > 0 else q
            for chunk in candidates:
                if chunk["vector"] is None:
                    continue
                scored.append((float(np.dot(q, chunk["vector"])), chunk))

    if not scored:  # lexical fallback (also used if embedding call failed)
        query_tokens = _tokenize(query_text)
        for chunk in candidates:
            scored.append((
                _lexical_score(query_tokens, chunk["text"], _rag_idf, _rag_corpus_size),
                chunk,
            ))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [(score, chunk) for score, chunk in scored[:top_k] if score > 0]
    if not top:
        return ""

    blocks = []
    for score, chunk in top:
        header = (f"--- {os.path.relpath(chunk['file'], WATCH_FOLDER)} "
                  f"(from line {chunk['start_line']}) ---")
        blocks.append(f"{header}\n{chunk['text']}")

    print(f"[RAG] Retrieved {len(top)} chunks for context "
          f"({', '.join(os.path.basename(c['file']) for _, c in top)})")

    return ("\n\nRelated code from elsewhere in this project. Use it only to understand "
            "definitions, signatures and types. Do NOT rewrite or output these:\n"
            + "\n\n".join(blocks))


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_lm_studio_fix(code: str, language: str, context: str = "") -> str:
    """Send code to local Gemma via LM Studio's OpenAI-compatible API, return corrected code."""
    system_prompt = (
        f"You are a code-fixing assistant. You will be given {language} code. "
        "Fix any syntax errors, compile errors, or obvious logic bugs. "
        "Return ONLY the corrected full code. "
        "Do not include explanations, comments about what you changed, or markdown code fences."
    )

    user_content = code

    if context:
        user_content += context

    response = requests.post(
        LM_STUDIO_URL,
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )

    # If LM Studio rejected the request, show WHY (the body has the real reason)
    if response.status_code != 200:
        print(f"LM Studio returned {response.status_code}. Body:")
        print(response.text)
        raise requests.exceptions.HTTPError(f"{response.status_code} Client Error")

    data = response.json()
    result_text = data["choices"][0]["message"]["content"]

    # Strip markdown fences if the model added them anyway
    cleaned = result_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Skip the language identifier line (e.g., ```python)
        lines = lines[1:]
        if lines and lines[-1].strip().endswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    return cleaned


def open_in_editor(filepath):
    if EDITOR_CLI_COMMAND and shutil.which(EDITOR_CLI_COMMAND):
        try:
            if sys.platform.startswith("win"):
                # Use shell=True for Windows CLI shortcuts like 'code' or 'cursor'
                subprocess.Popen(f'{EDITOR_CLI_COMMAND} -r "{filepath}"', shell=True)
            else:
                cli_path = shutil.which(EDITOR_CLI_COMMAND)
                subprocess.run([cli_path, "-r", filepath], check=False)
            return
        except Exception as e:
            print(f"(Could not open with '{EDITOR_CLI_COMMAND}' CLI: {e})")

    if EDITOR_APP_NAME is None:
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", EDITOR_APP_NAME, filepath], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(filepath)  # Cleaner native Windows opening
        else:
            subprocess.run(["xdg-open", filepath], check=False)
    except Exception as e:
        print(f"(Could not open editor automatically: {e})")


def fix_current_file():
    """Runs in a background thread: find the file, fix it, write it back."""
    global is_fixing
    is_fixing = True
    try:
        target = get_latest_code_file(WATCH_FOLDER)
        if target is None:
            print("No supported code file found to fix.")
            return

        ext = os.path.splitext(target)[1]
        language = LANGUAGE_MAP.get(ext, "code")
        print(f"Fixing: {target} ({language})")

        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            original_code = f.read()

        open_in_editor(os.path.abspath(target))

        build_rag_index(WATCH_FOLDER)
        context = retrieve_context(original_code, target)

        fixed_code = call_lm_studio_fix(original_code, language, context)

        if not fixed_code.strip():
            print("Model returned empty response, skipping write.")
            return

        if fixed_code.strip() == original_code.strip():
            print("[DEBUG] Model returned code identical to the original — nothing to write.")
            return

        print(f"[DEBUG] original={len(original_code)} chars, fixed={len(fixed_code)} chars")

        backup_path = target + ".bak"
        shutil.copy(target, backup_path)

        # Retry the write a few times in case Windows has a transient lock
        # (Defender/indexer/VS Code's watcher briefly holding the handle).
        last_err = None
        for attempt in range(5):
            try:
                with open(target, "w", encoding="utf-8", newline="") as f:
                    f.write(fixed_code)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                print(f"[DEBUG] write attempt {attempt+1} failed ({e}), retrying...")
                time_module.sleep(0.3)

        if last_err:
            raise last_err

        print(f"Fixed and saved: {target} (backup at {backup_path})")

    except requests.exceptions.HTTPError as e:
        print(f"\n--- FIX FAILED ---\nLM Studio API Error: {e}")
    except requests.exceptions.ConnectionError:
        print("\n--- FIX FAILED ---\nCould not reach LM Studio.")
    except Exception as e:
        import traceback
        print(f"\n--- FIX FAILED ---\nError while fixing file: {e}")
        traceback.print_exc()
    finally:
        is_fixing = False


def audio_callback(indata, frames, time, status):
    global last_trigger

    if status or is_fixing:
        return

    audio = indata[:, 0]
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-5:
        return
    crest_factor = peak / rms

    recent_peaks.append(peak)
    is_sudden_jump = (
        len(recent_peaks) >= 2
        and recent_peaks[-1] > recent_peaks[-2] * 2
    )

    now = time_module.time()
    if (
        peak > PEAK_THRESHOLD
        and crest_factor > TRANSIENT_RATIO
        and is_sudden_jump
        and (now - last_trigger) > COOLDOWN
    ):
        last_trigger = now
        print(f"👆 Snap detected! (Peak: {peak:.3f}, Sharpness: {crest_factor:.1f}) — starting fix...")
        threading.Thread(target=fix_current_file, daemon=True).start()


print("Calibrating background noise... Please stay quiet for 2 seconds.")
sd.sleep(2000)

# Warm the RAG index at startup so the first snap isn't slowed by indexing.
if RAG_ENABLED:
    print("Building project index for RAG...")
    build_rag_index(WATCH_FOLDER)

print(f"Listening for snaps! Watching folder: '{os.path.abspath(WATCH_FOLDER)}' (Ctrl+C to stop)")

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopped.")