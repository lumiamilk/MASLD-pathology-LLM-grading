"""
v2_eval_pipeline.py — Unified Dual-Condition Evaluation Pipeline v2
====================================================================
Condition A (Extraction): 病理描述_有診断 (full report with diagnostic conclusion)
Condition B (Inference):  病理描述_删除診断 (microscopy description only)

CLI:
  python v2_eval_pipeline.py [NUM_RUNS] [SAMPLE_LIMIT]    # Full evaluation
  python v2_eval_pipeline.py --smoke                       # Quick smoke test (2 samples per model)
  python v2_eval_pipeline.py --smoke --smoke-limit N        # Smoke test with N models
"""

import os, sys, subprocess, time, requests, json, re, datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

LLAMACPP_SERVER_EXE = r"D:\soft\to_run\ai\chatai\no_model\llama-b8826-bin-win-cuda-13.1-x64\llama-server.exe"
MODELS_DIR = r"Y:\models2"  # missing models (MoE, Hunyuan-4B, gpt-oss etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_THREADS = 16  # for local PC; use 36 on cloud

_IS_SMOKE = '--smoke' in sys.argv
_IS_RETHINK = '--rethink' in sys.argv  # re-test thinking models with 2048 max_tokens
if _IS_SMOKE:
    NUM_RUNS = 1
    SAMPLE_LIMIT = 0
else:
    NUM_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    SAMPLE_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

MAX_RETRIES = 3
BASE_TIMEOUT = 120
# Thinking models (DeepSeek-R1, Qwen3-Thinking, GLM-4, etc.) need ~1000-1500 tokens
# for reasoning + final answer. 512 was cutting them off mid-think.
# 2048 gives safe headroom; non-thinking models stop naturally at 1-5 tokens.
MAX_TOKENS = 2048

# =============================================================================
# 1. MODEL REGISTRY — maps model filename → metadata
# =============================================================================

MODEL_REGISTRY = {
    # --- Local, dense, non-thinking ---
    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf":    {"family": "qwen3",   "thinking": False, "parse": "default"},
    "Qwen3-4B-Q4_K_M.gguf":                  {"family": "qwen3",   "thinking": True,  "parse": "default"},
    "Qwen3-4B-Thinking-2507-Q4_K_M.gguf":    {"family": "qwen3",   "thinking": True,  "parse": "default"},
    "Qwen3-8B-Q4_K_M.gguf":                  {"family": "qwen3",   "thinking": True,  "parse": "default"},
    "gemma-3-4b-it-Q4_K_M.gguf":             {"family": "gemma",   "thinking": False, "parse": "default"},
    "gemma-3-12b-it-Q4_K_M.gguf":            {"family": "gemma",   "thinking": False, "parse": "default"},
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf":     {"family": "llama",   "thinking": False, "parse": "default"},
    "Llama-3.1-8B-Instruct-Q4_K_M.gguf":     {"family": "llama",   "thinking": False, "parse": "default"},
    "granite-3.3-8b-instruct-Q4_K_M.gguf":   {"family": "granite", "thinking": False, "parse": "default"},
    "medgemma-4b-it-Q4_K_M.gguf":            {"family": "gemma",   "thinking": False, "parse": "default"},
    "Hunyuan-4B-Instruct-Q4_K_M.gguf":       {"family": "hunyuan", "thinking": True,  "parse": "default"},
    "Hunyuan-7B-Instruct-Q4_K_M.gguf":       {"family": "hunyuan", "thinking": True,  "parse": "default"},
    # Filename variant
    "tencent_Hunyuan-7B-Instruct-Q4_K_M.gguf": {"family": "hunyuan", "thinking": True,  "parse": "default"},
    "tencent_Hunyuan-4B-Instruct-Q4_K_M.gguf": {"family": "hunyuan", "thinking": True,  "parse": "default"},

    # --- Local, MoE ---
    "Moonlight-16B-A3B-Instruct-Q4_K_M.gguf":          {"family": "moonlight","thinking": False, "parse": "default"},
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf":         {"family": "qwen3",   "thinking": False, "parse": "default"},
    "Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf":         {"family": "qwen3",   "thinking": True,  "parse": "default"},

    # --- Local, thinking, dense ---
    "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf":           {"family": "deepseek","thinking": True,  "parse": "default"},
    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf":         {"family": "deepseek","thinking": True,  "parse": "default"},  # old base, may work
    "GLM-4-9B-0414-Q4_K_M.gguf":                       {"family": "glm",     "thinking": True,  "parse": "default"},

    # --- Cloud, dense ---
    "Qwen3-14B-Q4_K_M.gguf":                           {"family": "qwen3",   "thinking": True,  "parse": "default"},
    "Qwen3-32B-Q4_K_M.gguf":                           {"family": "qwen3",   "thinking": True,  "parse": "default"},
    "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf":        {"family": "deepseek","thinking": True,  "parse": "default"},
    "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf":        {"family": "deepseek","thinking": True,  "parse": "default"},
    "GLM-4-32B-0414-Q4_K_M.gguf":                      {"family": "glm",     "thinking": True,  "parse": "default"},
    "gemma-3-27b-it-Q4_K_M.gguf":                      {"family": "gemma",   "thinking": False, "parse": "default"},
    "medgemma-27b-it-Q4_K_M.gguf":                     {"family": "gemma",   "thinking": False, "parse": "default"},

    # --- Special: GPT-OSS with reasoning levels ---
    "gpt-oss-20b-Q4_K_M.gguf":                         {"family": "gpt-oss", "thinking": True,  "parse": "gpt_oss",
                                                         "reasoning_modes": ["low", "medium", "high"]},
}

# =============================================================================
# 2. CONDITIONS AND TASKS
# =============================================================================

CONDITIONS = {
    'A_extraction': {
        'text_column': '病理描述_有诊断',
        'label': 'Condition A — Extraction (full report with diagnosis)'
    },
    'B_inference': {
        'text_column': '病理描述_删除诊断',
        'label': 'Condition B — Inference (description only, no diagnosis)'
    }
}

TASKS = {
    'fibrosis':      {'col': 'Fibrosis_Stage_0_4',    'name': '肝纤维化分期',     'labels': [0, 1, 2, 3, 4]},
    'inflammation':  {'col': 'Inflammation_Grade_0_4', 'name': '肝脏炎症分级',     'labels': [0, 1, 2, 3, 4]},
    'steatosis':     {'col': 'Steatosis_Grade_1_3',    'name': '肝脂肪变性分级',   'labels': [1, 2, 3]}
}

# =============================================================================
# 3. OUTPUT DIRECTORIES AND LOGGING
# =============================================================================

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', f'v2_eval_{ts}')
DETAILED_DIR = os.path.join(RESULTS_DIR, 'detailed_predictions')
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs', 'v2')

for d in [RESULTS_DIR, DETAILED_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f'v2_eval_{ts}.log')

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
        sys.stdout = self
    def write(self, m):
        self.terminal.write(m); self.log.write(m)
    def flush(self):
        self.terminal.flush(); self.log.flush()

Logger(LOG_FILE)
sys.stderr = sys.stdout

def log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

print("=" * 70)
print("  V2 UNIFIED DUAL-CONDITION EVALUATION PIPELINE")
print(f"  Models: {len(MODEL_REGISTRY)} registered")
print(f"  Runs: {NUM_RUNS}  |  Conditions: {len(CONDITIONS)}  |  Tasks: {len(TASKS)}")
print(f"  Samples: {'ALL' if SAMPLE_LIMIT==0 else SAMPLE_LIMIT}  |  Timestamp: {ts}")
print("=" * 70)

# =============================================================================
# 4. PROMPT AND PARSING
# =============================================================================

SYSTEM_PROMPT_TPL = (
    '你是一位顶尖的肝脏病理学专家。'
    '你的任务是根据病理描述对"{task_name}"进行分级。'
    '你的回答必须严格遵守规则：只输出一个属于 {labels_str} 的阿拉伯数字，'
    '禁止任何其他文字。'
)

# GPT-OSS reasoning mode prefixes
REASONING_PREFIXES = {
    "low":    "Reasoning: low",
    "medium": "Reasoning: medium",
    "high":   "Reasoning: high",
}

def build_system_prompt(task_name, labels_str, reasoning_mode=None):
    base = SYSTEM_PROMPT_TPL.format(task_name=task_name, labels_str=labels_str)
    if reasoning_mode and reasoning_mode in REASONING_PREFIXES:
        return f"{REASONING_PREFIXES[reasoning_mode]}\n\n{base}"
    return base

def build_few_shot_prompt(text, examples, task_name, labels_str):
    example_str = ""
    for i, ex in enumerate(examples):
        example_str += (
            f"[示例{i+1}]\n"
            f"病理描述：\n\"{ex['text']}\"\n"
            f"{task_name} ({labels_str}):\n{ex['label']}\n\n"
        )
    return (
        f"请参考以下示例，并对新的病理描述进行判断。\n\n"
        f"{example_str}"
        f"新的病理描述：\n\"{text}\""
    )

def parse_response(text, labels, parse_strategy="default"):
    """
    Unified response parser supporting all model output formats.

    parse_strategy:
      - "default": Handle <answer>, <think> tags, then extract number
      - "gpt_oss": Handle GPT-OSS channel markers, then extract number
    """
    if text == "LLM_CALL_ERROR" or text is None:
        return None

    # --- GPT-OSS channel marker parsing ---
    if parse_strategy == "gpt_oss":
        m = re.search(r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*)', text, re.S)
        if not m:
            m = re.search(r'<\|start\|>assistant(.*)', text, re.S)
        text = m.group(1) if m else text

    # --- Common logic ---
    # Priority 1: <answer>...</answer> tag (llama.cpp server may add for reasoning models)
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    else:
        # Priority 2: remove <think>...</think> blocks, keep post-think content
        cleaned = re.sub(r'<think>.*?</think>', '', text,
                         flags=re.DOTALL | re.IGNORECASE).strip()

    # Priority 3: extract standalone digits within valid range
    nums = re.findall(r'\b([0-4])\b', cleaned)
    if nums:
        val = int(nums[-1])  # take the LAST number (final answer after reasoning)
        return val if val in labels else None
    return None

# =============================================================================
# 5. LLAMA.CPP SERVER MANAGEMENT
# =============================================================================

API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
MODELS_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models"

def wait_for_server(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            hr = requests.get(HEALTH_URL, timeout=5)
            if hr.status_code == 200 and hr.json().get('status') == 'ok':
                mr = requests.get(MODELS_URL, timeout=5)
                if mr.status_code == 200 and len(mr.json().get('data', [])) > 0:
                    model_id = mr.json()['data'][0]['id']
                    log(f"Server ready — model: {model_id}")
                    return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            log(f"  health check: {e}")
        time.sleep(3)
    log("ERROR: Server timeout")
    return False

def start_server(model_path):
    # New llama.cpp b8826: --reasoning off keeps all output in content field
    # (prevents server from stripping think-tags into separate reasoning_content)
    cmd = [
        LLAMACPP_SERVER_EXE,
        "--threads", str(SERVER_THREADS),
        "--model", model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--reasoning", "off",
    ]
    log(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc

def stop_server(proc):
    if proc is None:
        return
    log("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log("Force killing server...")
        proc.kill()
        proc.wait()
    subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
    time.sleep(2)
    log("Server stopped.")

# =============================================================================
# 6. LLM API CALL
# =============================================================================

def call_llm(system_prompt, user_prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        timeout = BASE_TIMEOUT + (attempt - 1) * 60
        try:
            resp = requests.post(API_URL, headers=headers,
                                 data=json.dumps(payload), timeout=timeout)
            resp.raise_for_status()
            resp_json = resp.json()
            choice = resp_json['choices'][0]['message']
            # Only use content. reasoning_content contains think-tag text that
            # often contains medical numbers (F1, F2, etc.) from the reasoning
            # process — NOT the final answer. Using it would inflate parse errors.
            return (choice.get('content', '') or '').strip()
        except requests.exceptions.ReadTimeout:
            log(f"  [Retry {attempt}/{MAX_RETRIES}] ReadTimeout (timeout={timeout}s)")
            if attempt == MAX_RETRIES:
                return "LLM_CALL_ERROR"
            time.sleep(2 ** attempt + np.random.uniform(0, 1))
        except requests.exceptions.RequestException as e:
            log(f"  [Retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                return "LLM_CALL_ERROR"
            time.sleep(2 ** attempt + np.random.uniform(0, 1))
    return "LLM_CALL_ERROR"

# =============================================================================
# 7. METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, labels):
    n = len(y_true)
    if n == 0:
        return {}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mprec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    mrec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    per_class = {}
    for idx, lbl in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[str(lbl)] = {
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'specificity': round(spec, 4), 'f1': round(f1_c, 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'support': int(cm[idx, :].sum())
        }
    macro_spec = np.mean([per_class[str(l)]['specificity'] for l in labels])

    return {
        'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4),
        'weighted_f1': round(wf1, 4), 'macro_precision': round(mprec, 4),
        'macro_recall': round(mrec, 4), 'macro_specificity': round(macro_spec, 4),
        'per_class': per_class, 'confusion_matrix': cm.tolist(), 'n_valid': n
    }

# =============================================================================
# 8. MAIN EVALUATION LOOP
# =============================================================================

def run_evaluation():
    val_path = os.path.join(SCRIPT_DIR, 'processed_data', 'val_set.csv')
    train_path = os.path.join(SCRIPT_DIR, 'processed_data', 'train_set.csv')

    for p, name in [(val_path, 'val_set'), (train_path, 'train_set'),
                    (LLAMACPP_SERVER_EXE, 'llama-server.exe')]:
        if not os.path.exists(p):
            log(f"FATAL: {name} not found at {p}"); return

    val_df = pd.read_csv(val_path, encoding='utf-8-sig')
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    log(f"Loaded val={len(val_df)} rows, train={len(train_df)} rows")

    # --- Build effective model list ---
    # Find available .gguf files in MODELS_DIR
    available_models = {}
    if os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith('.gguf') and fname in MODEL_REGISTRY:
                available_models[fname] = MODEL_REGISTRY[fname]

    if not available_models:
        # Fallback: if MODELS_DIR not set up, look for models with full paths
        log("WARNING: No models found in MODELS_DIR. Looking for individual model files...")
        for fname, meta in MODEL_REGISTRY.items():
            candidate = os.path.join(MODELS_DIR, fname)
            if os.path.exists(candidate):
                available_models[fname] = meta

    log(f"Models available: {len(available_models)}")
    # --rethink filter: only thinking models
    if _IS_RETHINK:
        available_models = {k: v for k, v in available_models.items() if v.get('thinking')}
        log(f"--rethink: narrowed to {len(available_models)} thinking models")
    for name, meta in available_models.items():
        log(f"  - {name} (family={meta['family']}, thinking={meta['thinking']}, parse={meta['parse']})")

    if not available_models:
        log("FATAL: No models available. Check MODELS_DIR.")
        return

    all_summary = []
    all_predictions = []

    # Count total combos for progress
    total_combos = 0
    for fname, meta in available_models.items():
        reasoning_modes = meta.get('reasoning_modes', [None])
        total_combos += NUM_RUNS * len(CONDITIONS) * len(TASKS) * len(reasoning_modes)
    combo_idx = 0

    for run_num in range(1, NUM_RUNS + 1):
        log(f"\n{'#'*60}\n  RUN {run_num} / {NUM_RUNS}\n{'#'*60}")

        # Vary few-shot seed per run
        np.random.seed(42 + run_num * 100)
        shuffled_train = train_df.sample(frac=1, random_state=42 + run_num * 100).reset_index(drop=True)

        for model_fname, model_meta in available_models.items():
            model_path = os.path.join(MODELS_DIR, model_fname)
            parse_strategy = model_meta['parse']
            reasoning_modes = model_meta.get('reasoning_modes', [None])

            for reasoning_mode in reasoning_modes:
                # Build model display name
                if reasoning_mode:
                    model_display = f"{model_fname} ({reasoning_mode})"
                else:
                    model_display = model_fname

                log(f"\n  {'='*50}\n  Model: {model_display}\n  {'='*50}")

                # Start server for this model
                server_proc = None
                try:
                    server_proc = start_server(model_path)
                    time.sleep(2)
                    if not wait_for_server(timeout=300):
                        log("  ERROR: Server failed to start. Skipping model.")
                        continue

                    for cond_key, cond_cfg in CONDITIONS.items():
                        text_col = cond_cfg['text_column']
                        log(f"\n  --- {cond_cfg['label']} ---")

                        for task_key, task_cfg in TASKS.items():
                            combo_idx += 1
                            task_col = task_cfg['col']
                            task_name = task_cfg['name']
                            labels = task_cfg['labels']
                            labels_str = ", ".join(map(str, labels))

                            log(f"\n  [{combo_idx}/{total_combos}] run={run_num} {cond_key} {task_key} {model_display}")
                            t_start = time.time()

                            # Few-shot examples
                            few_shot = []
                            for lbl in sorted(labels):
                                subset = shuffled_train[shuffled_train[task_col] == lbl]
                                if len(subset) > 0:
                                    row = subset.iloc[0]
                                    few_shot.append({
                                        'text': str(row[text_col]),
                                        'label': lbl
                                    })

                            system_prompt = build_system_prompt(task_name, labels_str, reasoning_mode)

                            true_labels, pred_labels, sample_times = [], [], []
                            patient_ids = []
                            parse_errors = {'not_integer': 0, 'not_in_labels': 0, 'empty_output': 0, 'call_error': 0}

                            n_effective = SAMPLE_LIMIT if SAMPLE_LIMIT > 0 else len(val_df)
                            for i, (_, row) in enumerate(val_df.iterrows()):
                                if SAMPLE_LIMIT > 0 and i >= SAMPLE_LIMIT:
                                    break

                                patient_id = row['Patient_ID']
                                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                                true_label = row[task_col]
                                if pd.isna(true_label):
                                    continue

                                s_start = time.time()
                                user_prompt = build_few_shot_prompt(text, few_shot, task_name, labels_str)
                                response = call_llm(system_prompt, user_prompt)
                                pred = parse_response(response, labels, parse_strategy)
                                s_elapsed = time.time() - s_start

                                true_labels.append(int(true_label))
                                pred_labels.append(pred)
                                sample_times.append(s_elapsed)
                                patient_ids.append(str(patient_id))

                                # Classify parse errors
                                if pred is None:
                                    if response == "LLM_CALL_ERROR":
                                        parse_errors['call_error'] += 1
                                    elif response is None or response.strip() == "":
                                        parse_errors['empty_output'] += 1
                                    else:
                                        # Try to extract number to see if it's out-of-range
                                        nums = re.findall(r'\b(\d+)\b', response)
                                        if nums and int(nums[-1]) not in labels:
                                            parse_errors['not_in_labels'] += 1
                                        else:
                                            parse_errors['not_integer'] += 1

                                log(f"    [{i+1}/{n_effective}] true={int(true_label)} pred={pred} ({s_elapsed:.1f}s) | errs: {parse_errors}")

                            # Compute metrics
                            valid_idx = [i for i, p in enumerate(pred_labels) if p is not None]
                            y_true_valid = [true_labels[i] for i in valid_idx]
                            y_pred_valid = [pred_labels[i] for i in valid_idx]

                            metrics = compute_metrics(y_true_valid, y_pred_valid, labels)
                            error_rate = (len(pred_labels) - len(valid_idx)) / max(len(pred_labels), 1)
                            avg_time = np.mean(sample_times) if sample_times else 0
                            t_total = time.time() - t_start

                            summary = {
                                'run': run_num,
                                'condition': cond_key,
                                'task': task_key,
                                'model': model_display,
                                'model_family': model_meta['family'],
                                'thinking': model_meta['thinking'],
                                'parse_strategy': parse_strategy,
                                'reasoning_mode': reasoning_mode or '',
                                **{k: v for k, v in metrics.items() if k not in ('per_class', 'confusion_matrix')},
                                'error_rate': round(error_rate, 4),
                                'avg_time_s': round(avg_time, 2),
                                'total_time_s': round(t_total, 1),
                                'n_samples': len(pred_labels),
                                'n_valid': metrics.get('n_valid', 0),
                                **parse_errors,
                            }
                            for lbl_str, vals in metrics.get('per_class', {}).items():
                                for mn, mv in vals.items():
                                    summary[f'{mn}_class_{lbl_str}'] = mv

                            all_summary.append(summary)

                            for j in range(len(pred_labels)):
                                all_predictions.append({
                                    'run': run_num,
                                    'condition': cond_key,
                                    'task': task_key,
                                    'model': model_display,
                                    'model_family': model_meta['family'],
                                    'patient_id': patient_ids[j] if j < len(patient_ids) else '',
                                    'true_label': true_labels[j] if j < len(true_labels) else None,
                                    'predicted_label': pred_labels[j] if j < len(pred_labels) else None,
                                })

                            log(f"    Done. acc={metrics.get('accuracy')}, mf1={metrics.get('macro_f1')}, "
                                f"valid={metrics.get('n_valid')}/{len(pred_labels)}, time={t_total:.0f}s, "
                                f"errors={parse_errors}")

                            pd.DataFrame(all_summary).to_csv(
                                os.path.join(RESULTS_DIR, 'v2_summary_all.csv'),
                                index=False, encoding='utf-8-sig')
                            pd.DataFrame(all_predictions).to_csv(
                                os.path.join(RESULTS_DIR, 'v2_predictions_all.csv'),
                                index=False, encoding='utf-8-sig')

                finally:
                    if server_proc:
                        stop_server(server_proc)

    # =========================================================================
    # 9. FINAL AGGREGATION
    # =========================================================================
    log(f"\n{'='*60}\n  EVALUATION COMPLETE — GENERATING REPORTS\n{'='*60}")

    df_summary = pd.DataFrame(all_summary)
    df_preds = pd.DataFrame(all_predictions)

    agg_cols = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision',
                'macro_recall', 'macro_specificity', 'error_rate', 'avg_time_s']

    agg = df_summary.groupby(['condition', 'task', 'model']).agg(
        **{f'{c}_mean': (c, 'mean') for c in agg_cols},
        **{f'{c}_std': (c, 'std') for c in agg_cols},
        n_runs=('run', 'count')
    ).reset_index()

    agg.to_csv(os.path.join(RESULTS_DIR, 'v2_aggregated_summary.csv'),
               index=False, encoding='utf-8-sig')

    log("\n--- AGGREGATED RESULTS ---")
    log(agg.to_string())

    # Condition comparison
    log("\n--- CONDITION COMPARISON (A vs B gap) ---")
    for task_key in TASKS:
        log(f"\n  Task: {task_key}")
        a_rows = df_summary[(df_summary['condition'] == 'A_extraction') & (df_summary['task'] == task_key)]
        b_rows = df_summary[(df_summary['condition'] == 'B_inference') & (df_summary['task'] == task_key)]
        for metric in ['macro_f1', 'accuracy']:
            if len(a_rows) > 0 and len(b_rows) > 0:
                gap = a_rows[metric].mean() - b_rows[metric].mean()
                log(f"    {metric}: A={a_rows[metric].mean():.4f}  B={b_rows[metric].mean():.4f}  gap={gap:+.4f}")

    # Parse error summary
    log("\n--- PARSE ERROR SUMMARY ---")
    for model_name in df_summary['model'].unique():
        model_rows = df_summary[df_summary['model'] == model_name]
        total_n = model_rows['n_samples'].sum()
        for err_type in ['not_integer', 'not_in_labels', 'empty_output', 'call_error']:
            if err_type in model_rows.columns:
                err_count = model_rows[err_type].sum()
                if err_count > 0:
                    log(f"  {model_name}: {err_type}={err_count}/{total_n} ({100*err_count/total_n:.1f}%)")

    log(f"\nAll results saved to: {RESULTS_DIR}")

    return df_summary, df_preds

# =============================================================================
# 10. SMOKE TEST (uses SAME call_llm/parse_response/server functions as eval)
# =============================================================================

def run_smoke_test(max_models=None):
    """Quick smoke test: 2 samples per model, validates the actual pipeline code path."""
    subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
    time.sleep(2)

    log(f"\n{'='*60}")
    log("  SMOKE TEST — validating pipeline code path on all models")
    log(f"{'='*60}")

    val_path = os.path.join(SCRIPT_DIR, 'processed_data', 'val_set.csv')
    train_path = os.path.join(SCRIPT_DIR, 'processed_data', 'train_set.csv')
    if not os.path.exists(val_path):
        log("FATAL: val_set.csv not found"); return

    val_df = pd.read_csv(val_path, encoding='utf-8-sig')
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')

    # Find available models
    available = {}
    for fname, meta in MODEL_REGISTRY.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            # Skip translation models
            if 'Galtransl' in fname:
                log(f"  SKIP {fname} — translation model")
                continue
            available[fname] = meta

    if max_models:
        available = dict(list(available.items())[:max_models])

    log(f"Testing {len(available)} models with 2 samples each...\n")

    # Test samples from validation set
    from sklearn.metrics import f1_score, accuracy_score
    test_samples = val_df.sample(min(2, len(val_df)), random_state=42)
    labels_fib = [0, 1, 2, 3, 4]
    labels_str_fib = "0, 1, 2, 3, 4"

    results = []
    for model_fname, model_meta in available.items():
        model_path = os.path.join(MODELS_DIR, model_fname)
        parse_strategy = model_meta['parse']
        t0 = time.time()

        log(f"\n{'-'*50}")
        log(f"  {model_fname} (family={model_meta['family']}, thinking={model_meta['thinking']})")

        server_proc = None
        issues = []
        try:
            server_proc = start_server(model_path)
            time.sleep(2)
            if not wait_for_server(timeout=180):
                issues.append("SERVER_FAIL")
                log("  [FAIL] Server failed to start")
                results.append({'model': model_fname, 'status': 'FAIL', 'issues': '; '.join(issues), 'elapsed_s': round(time.time()-t0,1)})
                continue

            # Build few-shot from train set (same logic as pipeline)
            system_prompt = build_system_prompt("肝纤维化分期", labels_str_fib)

            all_ok = True
            for i, (_, row) in enumerate(test_samples.iterrows()):
                text_col = '病理描述_有诊断'  # Use Condition A text for smoke test
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                true_label = int(row['Fibrosis_Stage_0_4']) if pd.notna(row['Fibrosis_Stage_0_4']) else None

                # Minimal few-shot example
                few_shot = []
                for lbl in sorted(labels_fib):
                    subset = train_df[train_df['Fibrosis_Stage_0_4'] == lbl]
                    if len(subset) > 0:
                        few_shot.append({'text': str(subset.iloc[0][text_col]), 'label': lbl})

                user_prompt = build_few_shot_prompt(text, few_shot, "肝纤维化分期", labels_str_fib)

                # === USE THE EXACT SAME call_llm as the evaluation pipeline ===
                response = call_llm(system_prompt, user_prompt)
                pred = parse_response(response, labels_fib, parse_strategy)

                token_count = len(response.split()) if response and response != "LLM_CALL_ERROR" else 0
                raw_preview = (response or '(empty)')[:150].replace('\n', '\\n') if response else '(empty)'

                if pred is None:
                    issues.append(f"S{i+1}:PARSE_FAIL")
                    all_ok = False
                    log(f"  Sample {i+1}: pred=None | true={true_label} | raw: {raw_preview}")
                else:
                    log(f"  Sample {i+1}: pred={pred} | true={true_label} | tokens={token_count}")

                if token_count > 512:
                    issues.append(f"S{i+1}:LONG_OUTPUT({token_count} tokens)")
                    log(f"  [WARN] Long output: {token_count} tokens")

            status = 'PASS' if all_ok and not issues else 'WARN' if all_ok else 'FAIL'
            results.append({'model': model_fname, 'status': status, 'issues': '; '.join(issues) if issues else 'none', 'elapsed_s': round(time.time()-t0,1)})
            log(f"  {'[OK]' if status=='PASS' else '[WARN]' if status=='WARN' else '[FAIL]'} {status}")

        except Exception as e:
            log(f"  [FAIL] CRASH: {e}")
            import traceback; traceback.print_exc()
            results.append({'model': model_fname, 'status': 'CRASH', 'issues': str(e), 'elapsed_s': round(time.time()-t0,1)})
        finally:
            if server_proc:
                stop_server(server_proc)
            time.sleep(2)

    # Summary
    log(f"\n{'='*60}")
    log(f"  SMOKE TEST SUMMARY")
    log(f"{'='*60}")
    pass_count = sum(1 for r in results if r['status'] == 'PASS')
    warn_count = sum(1 for r in results if r['status'] == 'WARN')
    fail_count = sum(1 for r in results if r['status'] in ('FAIL', 'CRASH'))
    log(f"  [OK] PASS: {pass_count}  [WARN] WARN: {warn_count}  [FAIL] FAIL: {fail_count}")

    for r in results:
        icon = '[OK]' if r['status'] == 'PASS' else ('[WARN]' if r['status'] == 'WARN' else '[FAIL]')
        log(f"  {icon} {r['model']} ({r['elapsed_s']}s)")
        if r['issues'] != 'none':
            log(f"      Issues: {r['issues']}")

    # Save results
    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, 'v2_smoke_results.csv'),
        index=False, encoding='utf-8-sig')
    log(f"\nResults: {RESULTS_DIR}/v2_smoke_results.csv")
    log("Smoke test complete.")

# =============================================================================
# 11. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    server_proc = None
    try:
        if _IS_SMOKE:
            smoke_limit = None
            for i, arg in enumerate(sys.argv):
                if arg == '--smoke-limit' and i + 1 < len(sys.argv):
                    smoke_limit = int(sys.argv[i + 1])
            run_smoke_test(max_models=smoke_limit)
        else:
            run_evaluation()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
        log("\nPipeline finished.")
