import os
import re
import json
import time
import math
import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import pandas as pd
import streamlit as st
import graphviz
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("radiology_workbench_production")

APP_TITLE = "AI Radiology Workbench Pro"
APP_VERSION = "3.3.0" # Incremented for Co-occurrence Smart Suggestions

# --- CONFIGURATION ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
OLLAMA_PARSE_MODEL = os.getenv("OLLAMA_PARSE_MODEL", OLLAMA_MODEL)
OLLAMA_REPORT_MODEL = os.getenv("OLLAMA_REPORT_MODEL", OLLAMA_MODEL)
OLLAMA_QA_MODEL = os.getenv("OLLAMA_QA_MODEL", OLLAMA_MODEL)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_FAST = int(os.getenv("OLLAMA_TIMEOUT_FAST", "12"))
OLLAMA_TIMEOUT_SLOW = int(os.getenv("OLLAMA_TIMEOUT_SLOW", "180"))

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "12345678")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "25"))
MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "15"))
MAX_AUTOCOMPLETE_TERMS = int(os.getenv("MAX_AUTOCOMPLETE_TERMS", "1500"))
ENABLE_VECTOR_RERANK = os.getenv("ENABLE_VECTOR_RERANK", "false").lower() == "true"
VECTOR_RERANK_TOPN = int(os.getenv("VECTOR_RERANK_TOPN", "50"))

# --- DICTIONARIES & MAPPINGS ---
HEADER_ALIASES = {
    "History": [r"HISTORY", r"CLINICAL INFORMATION", r"CLINICAL INSTRUCTION", r"CLINICAL HISTORY"],
    "Technique": [r"TECHNIQUE", r"TECHNIQUES"],
    "Comparison": [r"COMPARISON", r"COMPARED WITH DATE"],
    "Findings": [r"FINDINGS?", r"LUNGS? AND PLEURA?", r"HEART AND MEDIASTINUM", r"BONY THORAX", r"BONES AND JOINTS", r"SOFT TISSUES"],
    "Impression": [r"IMPRESSION", r"IMP"],
    "Recommendation": [r"RECOMMENDATION", r"PLAN", r"RECOMMENDATIONS"],
}

KNOWN_MODALITIES = {
    "us": ["us", "ultrasound", "sonography"],
    "ct": ["ct", "computed tomography"],
    "mri": ["mri", "mr", "magnetic resonance"],
    "xray": ["xray", "x-ray", "xr", "radiograph"],
    "pet": ["pet", "pet-ct", "nuclear"],
    "mammo": ["mammo", "mammography", "mammogram"],
    "fluoro": ["fluoroscopy", "barium", "hsg"],
}

KNOWN_ORGANS = [
    "thyroid", "right thyroid lobe", "left thyroid lobe", "isthmus", "cervical lymph nodes", "lymph nodes",
    "liver", "gallbladder", "pancreas", "spleen", "kidneys", "kidney", "right kidney", "left kidney",
    "adrenals", "adrenal", "right adrenal", "left adrenal", "bladder", "uterus", "ovaries", "prostate",
    "testes", "bowel loops", "appendix", "colon", "rectum", "vasculature", "aorta", "inferior vena cava",
    "portal vein", "lungs", "lung", "right lung", "left lung", "pleura", "diaphragm", "heart", "pericardium",
    "mediastinum", "hilar structures", "trachea", "esophagus", "spine", "vertebrae", "sacrum", "ribs",
    "sternum", "breast", "axillary lymph nodes", "brain", "cerebellum", "brainstem", "ventricles", "pituitary",
    "parathyroid"
]

KNOWN_FINDINGS = [
    "nodule", "cyst", "microcalcifications", "calcification", "enlarged", "lesion", "mass", "tumor",
    "hypoechoic", "hyperechoic", "isoechoic", "anechoic", "heterogeneous", "homogeneous", "normal",
    "unremarkable", "effusion", "ascites", "pleural effusion", "consolidation", "atelectasis", "pneumothorax",
    "fracture", "dislocation", "osteoporosis", "stenosis", "occlusion", "thrombosis", "edema", "infarct",
    "hemorrhage", "metastasis", "lymphadenopathy", "fibrosis", "cirrhosis", "steatosis", "hydronephrosis",
    "nephrolithiasis", "cholelithiasis", "splenomegaly", "hepatomegaly"
]

SEVERITY_LEVELS = ["critical", "urgent", "severe", "moderate", "mild", "normal", "benign", "malignant"]
STOPWORDS = {
    "show", "find", "for", "with", "and", "the", "a", "an", "of", "in", "on", "patient", "all", "any",
    "from", "to", "by", "at", "as", "is", "are", "was", "were", "has", "have", "had", "this", "that",
    "these", "those", "what", "which", "who", "how", "can", "could", "should", "would"
}

STUDY_TEMPLATES = {
    "abdomenus": {"organs": ["liver", "gallbladder", "spleen", "pancreas", "right kidney", "left kidney", "right adrenal", "left adrenal", "bladder", "bowel loops", "vasculature"], "default_text": "Unremarkable.", "exam_label": "Ultrasound Abdomen"},
    "chestxr": {"organs": ["right lung", "left lung", "pleura", "heart", "mediastinum", "hilar structures", "bony thorax"], "default_text": "Unremarkable.", "exam_label": "Chest X-Ray"},
    "thyroidus": {"organs": ["right thyroid lobe", "left thyroid lobe", "isthmus", "cervical lymph nodes"], "default_text": "No focal abnormality identified.", "exam_label": "Ultrasound Thyroid"},
    "pelvisus": {"organs": ["uterus", "ovaries", "bladder", "bowel loops"], "default_text": "Unremarkable.", "exam_label": "Ultrasound Pelvis"},
    "breastus": {"organs": ["breast", "axillary lymph nodes"], "default_text": "No suspicious lesion identified.", "exam_label": "Ultrasound Breast"},
    "chestct": {"organs": ["right lung", "left lung", "pleura", "heart", "pericardium", "mediastinum", "hilar structures", "esophagus", "trachea", "spine", "bony thorax"], "default_text": "Unremarkable.", "exam_label": "CT Chest"},
    "abdomenct": {"organs": ["liver", "gallbladder", "spleen", "pancreas", "right kidney", "left kidney", "right adrenal", "left adrenal", "bladder", "bowel loops", "aorta", "inferior vena cava"], "default_text": "Unremarkable.", "exam_label": "CT Abdomen"},
}

REPORT_TEMPLATES = {
    "Ultrasound Thyroid": "ULTRASOUND THYROID / NECK\nDate: {date}\nPatient ID: {patient_id}\nGender: {gender}\nAge: {age}\n\nTECHNIQUE:\nHigh-frequency grayscale and Doppler ultrasound of the thyroid gland and neck.\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}\n\nRECOMMENDATIONS:\n{recommendations}\n\nReporting Radiologist",
    "Ultrasound Abdomen": "ULTRASOUND WHOLE ABDOMEN\nDate: {date}\nPatient ID: {patient_id}\nGender: {gender}\nAge: {age}\n\nTECHNIQUE:\nGrayscale and Doppler ultrasound examination of the abdomen.\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}\n\nRECOMMENDATIONS:\n{recommendations}\n\nReporting Radiologist",
    "CT Chest": "CT CHEST\nDate: {date}\nPatient ID: {patient_id}\nGender: {gender}\nAge: {age}\n\nTECHNIQUE:\n{technique}\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}\n\nRECOMMENDATIONS:\n{recommendations}\n\nReporting Radiologist",
    "General": "RADIOLOGY REPORT\nDate: {date}\nPatient ID: {patient_id}\nExam: {exam}\nGender: {gender}\nAge: {age}\n\nCLINICAL HISTORY:\n{history}\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}\n\nRECOMMENDATIONS:\n{recommendations}\n\nReporting Radiologist",
}

QA_SYSTEM_PROMPTS = {
    "General Radiologist": "You are a board-certified radiologist with 20 years of experience. Answer concisely, accurately, and only from the provided evidence.",
    "Consultant Radiologist": "You are a senior consultant radiologist. Provide a focused, evidence-based answer with differential diagnoses and guideline-oriented recommendations when evidence supports them.",
    "Teaching Mode": "You are a radiology attending teaching a resident. Explain stepwise, highlight key imaging features, and stay grounded in provided evidence.",
    "Patient-Friendly": "You are a radiologist explaining findings to a patient. Use simple, clear language. Avoid jargon and be accurate.",
    "Urgent / Emergency": "You are an emergency radiologist. Put critical findings first, state urgency clearly, and recommend next actions only when supported by evidence.",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --bg-primary:#0b1220; --bg-secondary:#111827; --bg-card:#0f172a; --bg-card-2:#131d33;
  --accent:#26c6da; --accent-2:#60a5fa; --text:#e5edf6; --muted:#94a3b8; --border:#23314d;
  --success:#22c55e; --warn:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp { background: radial-gradient(circle at top right, rgba(38,198,218,.12), transparent 20%), linear-gradient(180deg, #09101d 0%, #0b1220 100%); color: var(--text); }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }
.main-header { background: linear-gradient(135deg, rgba(38,198,218,.16), rgba(96,165,250,.08)); border: 1px solid var(--border); border-radius: 18px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 10px 40px rgba(2, 8, 23, .35); }
.main-header h1 { font-family: "IBM Plex Mono", monospace; color: var(--accent); font-size: 1.5rem; margin: 0; }
.main-header p { color: var(--muted); margin: 6px 0 0 0; font-size: .92rem; }
.status-badge { display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px; font:600 .72rem "IBM Plex Mono", monospace; border:1px solid var(--border); }
.badge-ok{background:rgba(34,197,94,.12); color:#86efac;} .badge-err{background:rgba(239,68,68,.12); color:#fca5a5;} .badge-warn{background:rgba(245,158,11,.12); color:#fcd34d;}
.metric-row { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-bottom: 14px; }
.metric-box { background: linear-gradient(180deg, rgba(17,24,39,.85), rgba(15,23,42,.92)); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }
.metric-val { font: 700 1.45rem "IBM Plex Mono", monospace; color: var(--accent); } .metric-lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.result-card { background: linear-gradient(180deg, rgba(15,23,42,.95), rgba(19,29,51,.88)); border:1px solid var(--border); border-radius:16px; padding:16px; margin-bottom:12px; }
.result-card:hover { border-color: rgba(38,198,218,.45); }
.card-header { display:flex; justify-content:space-between; gap:8px; margin-bottom:8px; flex-wrap:wrap; }
.patient-id { font: 700 .92rem "IBM Plex Mono", monospace; color: var(--accent); }
.modality-tag { font:600 .72rem "IBM Plex Mono", monospace; padding:3px 8px; border-radius:999px; border:1px solid rgba(96,165,250,.35); color:#bfdbfe; background:rgba(96,165,250,.10); }
.report-output { background: rgba(15,23,42,.96); border:1px solid rgba(38,198,218,.40); border-radius:16px; padding:18px 20px; font: .84rem/1.75 "IBM Plex Mono", monospace; white-space: pre-wrap; }
.chat-user { background: rgba(96,165,250,.10); border:1px solid rgba(96,165,250,.20); border-radius:14px 14px 4px 14px; padding:10px 12px; margin:6px 0 6px 46px; }
.chat-ai { background: rgba(15,23,42,.9); border:1px solid var(--border); border-radius:14px 14px 14px 4px; padding:10px 12px; margin:6px 46px 6px 0; }
.small-muted { color: var(--muted); font-size: .82rem; }
.stButton>button[kind="secondary"] { border-radius: 8px; padding: 4px 10px; font-size: 0.8rem; background: rgba(38,198,218,0.1); border: 1px solid var(--border); color: var(--accent); }
</style>
"""

# --- UTILS ---
def normalize_text(value):
    if value is None: return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())

def unique_preserve_order(values):
    seen, out = set(), []
    for v in values:
        if v is None: continue
        s = str(v).strip()
        key = normalize_text(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out

def safe_int(value, default=0):
    try: return default if value in (None, "") else int(float(value))
    except Exception: return default

def safe_float(value, default=None):
    try: return default if value in (None, "") else float(value)
    except Exception: return default

def truncate_text(text, max_len=500):
    t = str(text or "").strip()
    return t if len(t) <= max_len else t[:max_len - 3] + "..."

def parse_size_to_cm(text):
    if not text: return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)", text, re.I)
    if not m: return None
    value = float(m.group(1))
    return value / 10.0 if m.group(2).lower() == "mm" else value

def extract_measurements_from_text(text):
    return [m.strip() for m in re.findall(r"\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*){0,2}(?:cm|mm)", text or "", re.I)]

def extract_organs_from_text(text):
    tn = normalize_text(text)
    return unique_preserve_order([organ for organ in sorted(KNOWN_ORGANS, key=len, reverse=True) if normalize_text(organ) in tn])

def modality_to_study_bucket(modality, exam):
    combined = normalize_text(f"{modality} {exam}")
    if "thyroid" in combined or ("neck" in combined and "us" in combined): return "thyroidus"
    if "breast" in combined and "us" in combined: return "breastus"
    if "pelvis" in combined and "us" in combined: return "pelvisus"
    if any(k in combined for k in ["abdomen", "hepato", "liver", "gallbladder", "pancrea", "renal"]):
        return "abdomenct" if "ct" in combined else "abdomenus"
    if any(k in combined for k in ["chest", "lung", "pleura", "thorax"]):
        return "chestct" if "ct" in combined else "chestxr"
    return None

def severity_rank(severity):
    order = {"critical": 0, "urgent": 1, "malignant": 1, "severe": 2, "moderate": 3, "mild": 4, "normal": 5, "benign": 5}
    return order.get(normalize_text(severity), 6)

@dataclass
class SearchResult:
    hn: str = "NA"
    gender: str = "?"
    age: Any = "?"
    studyid: str = ""
    studydate: str = ""
    severity: str = ""
    clinicalinstruction: str = ""
    modality: str = ""
    examname: str = ""
    findingtext: str = ""
    organs: List[str] = field(default_factory=list)
    maxdimensioncm: Optional[float] = None
    score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0
    def as_dict(self): return self.__dict__.copy()

# --- CLIENTS ---
class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    def _post(self, path, payload, timeout):
        r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    @st.cache_data(ttl=30)
    def list_models(_self):
        try:
            data = _self._post("/api/tags", {}, 10)
            return sorted([m.get("name", "") for m in data.get("models", []) if m.get("name")])
        except Exception: return []
    def is_ready(self, model):
        try:
            self._post("/api/generate", {"model": model, "prompt": "hi", "stream": False}, 8)
            return True, f"Ollama OK: {model}"
        except Exception as e: return False, str(e)
    def generate(self, prompt, model, timeout, fmt=None, temperature=0.1):
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": 8192}}
        if fmt: payload["format"] = fmt
        data = self._post("/api/generate", payload, timeout)
        return (data.get("response") or "").strip()
    @st.cache_data(ttl=300, show_spinner=False)
    def embed_cached(_self, texts, model):
        out = []
        for text in texts:
            try:
                data = _self._post("/api/embeddings", {"model": model, "prompt": text}, 40)
                out.append(data.get("embedding") or [])
            except Exception: out.append([])
        return out

class QueryParserLLM:
    def __init__(self, ollama):
        self.ollama = ollama
    @staticmethod
    def _regex_parse(query):
        q = normalize_text(query)
        out = {"keywords": [w for w in re.findall(r"[a-z]{3,}", q) if w not in STOPWORDS][:8]}
        if "female" in q or "woman" in q: out["gender"] = "Female"
        elif "male" in q or "man" in q: out["gender"] = "Male"
        age_range = re.search(r"(\d{1,3})\s*[-to]{1,3}\s*(\d{1,3})\s*(?:years|yrs|y)?", q)
        if age_range:
            a, b = sorted([int(age_range.group(1)), int(age_range.group(2))])
            out["minage"], out["maxage"] = a, b
        single_age = re.search(r"(?:age|aged)\s*(\d{1,3})", q)
        if single_age and "minage" not in out:
            out["minage"] = int(single_age.group(1))
            out["maxage"] = int(single_age.group(1))
        for canon, aliases in KNOWN_MODALITIES.items():
            if any(a in q for a in aliases):
                out["modality"] = canon
                break
        for organ in sorted(KNOWN_ORGANS, key=len, reverse=True):
            if normalize_text(organ) in q:
                out["targetorgan"] = organ
                break
        for finding in sorted(KNOWN_FINDINGS, key=len, reverse=True):
            if normalize_text(finding) in q:
                out["finding"] = finding
                break
        dim = parse_size_to_cm(query)
        if dim is not None: out["mindimensioncm"] = dim
        for sev in SEVERITY_LEVELS:
            if sev in q:
                out["severity"] = sev
                break
        return out
    @st.cache_data(ttl=600, show_spinner=False)
    def parse_query(_self, query):
        if not query.strip(): return {}
        prompt = f"You are a medical NLP extraction API for radiology search. Return ONLY valid JSON. Omit absent keys. Convert mm to cm. Allowed keys: gender, minage, maxage, modality, examname, targetorgan, finding, mindimensioncm, maxdimensioncm, severity, keywords. Query: {query} JSON:"
        try:
            raw = _self.ollama.generate(prompt, OLLAMA_PARSE_MODEL, OLLAMA_TIMEOUT_FAST, fmt="json", temperature=0)
            parsed = json.loads(raw)
            regex = _self._regex_parse(query)
            merged = {**regex, **parsed}
            merged["keywords"] = unique_preserve_order((regex.get("keywords") or []) + (parsed.get("keywords") or []))[:10]
            return merged
        except Exception: return _self._regex_parse(query)

class GraphBuilder:
    def __init__(self, uri, user, password, database):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=3600, max_connection_pool_size=50, connection_timeout=15)

    def verify(self):
        try:
            with self.driver.session(database=self.database) as s:
                rec = s.run("RETURN 1 AS ok").single()
                return (True, "Neo4j connected") if rec and rec["ok"] == 1 else (False, "No response")
        except Exception as e: return False, str(e)

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_filter_options(_self, filters_key=""):
        filters = json.loads(filters_key) if filters_key else {}
        modality = normalize_text(filters.get("modality"))
        severity = normalize_text(filters.get("severity"))
        def fetch(cypher, params):
            try:
                with _self.driver.session(database=_self.database) as s:
                    rec = s.run(cypher, params).single()
                    return [x for x in (rec["vals"] if rec else []) if x]
            except Exception: return []
        if not modality and not severity:
            queries = {
                "exams": "MATCH (e:Exam) RETURN collect(DISTINCT e.name)[0..100] AS vals",
                "modalities": "MATCH (m:Modality) RETURN collect(DISTINCT coalesce(m.name,m.type))[0..100] AS vals",
                "severity": "MATCH (s:Study) WHERE s.severity IS NOT NULL RETURN collect(DISTINCT s.severity)[0..100] AS vals",
            }
            return {k: unique_preserve_order(fetch(q, {})) for k, q in queries.items()}
        cascade = {
            "exams": "MATCH (s:Study)-[:HAS_EXAM]->(e:Exam) OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality) WHERE ($modality='' OR toLower(coalesce(m.name,m.type,'')) CONTAINS $modality) AND ($severity='' OR toLower(coalesce(s.severity,'')) CONTAINS $severity) RETURN collect(DISTINCT e.name)[0..100] AS vals",
            "modalities": "MATCH (s:Study)-[:HAS_MODALITY]->(m:Modality) WHERE ($severity='' OR toLower(coalesce(s.severity,'')) CONTAINS $severity) RETURN collect(DISTINCT coalesce(m.name,m.type))[0..100] AS vals",
            "severity": "MATCH (s:Study)-[:HAS_MODALITY]->(m:Modality) WHERE ($modality='' OR toLower(coalesce(m.name,m.type,'')) CONTAINS $modality) AND s.severity IS NOT NULL RETURN collect(DISTINCT s.severity)[0..100] AS vals",
        }
        params = {"modality": modality, "severity": severity}
        return {k: unique_preserve_order(fetch(q, params)) for k, q in cascade.items()}

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_autocomplete_terms(_self, limit=MAX_AUTOCOMPLETE_TERMS):
        query = """
        CALL {
          MATCH (ft:FindingTerm) RETURN DISTINCT 'Finding: ' + ft.name AS term LIMIT $limit
          UNION
          MATCH (o:Organ) RETURN DISTINCT 'Organ: ' + o.name AS term LIMIT $limit
          UNION
          MATCH (m:Modality) RETURN DISTINCT 'Modality: ' + coalesce(m.name,m.type) AS term LIMIT $limit
          UNION
          MATCH (e:Exam) RETURN DISTINCT 'Exam: ' + e.name AS term LIMIT $limit
        }
        RETURN term ORDER BY term
        """
        try:
            with _self.driver.session(database=_self.database) as s:
                return [r["term"] for r in s.run(query, {"limit": limit}) if r["term"]]
        except Exception: return []
        
    @st.cache_data(ttl=60, show_spinner=False)
    def get_dynamic_autocomplete_terms(_self, selected_terms_tuple: tuple, limit=MAX_AUTOCOMPLETE_TERMS):
        """v3.3: Traverses up to 3 hops through Studies to find actual clinical co-occurrences."""
        base_terms = _self.get_autocomplete_terms(limit)
        if not selected_terms_tuple:
            return base_terms
        
        names = [t.split(": ", 1)[1] if ": " in t else t for t in selected_terms_tuple]
        
        # Traverse 1-to-3 hops but prevent crossing over Patient nodes to keep scope within clinical studies
        cypher = """
        MATCH (n)
        WHERE n.name IN $names OR coalesce(n.type, n.name) IN $names
        MATCH path = (n)-[*1..3]-(m)
        WHERE (m:Organ OR m:Finding OR m:FindingTerm OR m:Modality OR m:Exam) 
          AND m <> n
          AND NONE(x IN nodes(path) WHERE x:Patient)
        WITH m, count(*) AS weight
        ORDER BY weight DESC
        LIMIT 50
        WITH DISTINCT labels(m)[0] AS raw_lbl, coalesce(m.name, m.type, 'Unknown') AS val
        WITH CASE raw_lbl WHEN 'FindingTerm' THEN 'Finding' ELSE raw_lbl END AS lbl, val
        WHERE lbl IN ['Organ', 'Finding', 'Modality', 'Exam']
        RETURN lbl + ': ' + val AS term
        """
        related = []
        try:
            with _self.driver.session(database=_self.database) as s:
                related = [r["term"] for r in s.run(cypher, {"names": names}) if r["term"]]
        except Exception as e:
            logger.warning(f"Co-occurrence autocomplete failed: {e}")
            pass
            
        final_list = list(selected_terms_tuple)
        seen = set(final_list)
        
        for t in related:
            if t not in seen:
                final_list.append(t)
                seen.add(t)
                
        for t in base_terms:
            if t not in seen:
                final_list.append(t)
                seen.add(t)
                
        return final_list

    @st.cache_data(ttl=300, show_spinner=False)
    def get_subgraph_for_terms(_self, terms: List[str]):
        """v3.3: Visualizes the 1-to-3 hop co-occurrence relations rather than strict ontology links."""
        if not terms: return [], []
        names = [t.split(": ", 1)[1] if ": " in t else t for t in terms]
        cypher = """
        MATCH (n)
        WHERE n.name IN $names OR coalesce(n.type, n.name) IN $names
        MATCH path = (n)-[*1..3]-(m)
        WHERE (m:Organ OR m:Finding OR m:FindingTerm OR m:Modality OR m:Exam) 
          AND m <> n
          AND NONE(x IN nodes(path) WHERE x:Patient)
        WITH n, m, count(*) AS weight
        ORDER BY weight DESC
        LIMIT 30
        RETURN labels(n)[0] as raw_l1, coalesce(n.name, n.type, 'Unknown') as n1, 'CO-OCCURS' as rel, 
               labels(m)[0] as raw_l2, coalesce(m.name, m.type, 'Unknown') as n2
        """
        nodes, edges = set(), []
        try:
            with _self.driver.session(database=_self.database) as s:
                for r in s.run(cypher, {"names": names}):
                    # Format node 1
                    n1 = str(r['n1']) if r['n1'] is not None else 'Unknown'
                    raw_l1 = str(r['raw_l1']) if r['raw_l1'] is not None else 'Unknown'
                    l1 = 'Finding' if raw_l1 == 'FindingTerm' else raw_l1
                    
                    # Format node 2
                    n2 = str(r['n2']) if r['n2'] is not None else 'Unknown'
                    raw_l2 = str(r['raw_l2']) if r['raw_l2'] is not None else 'Unknown'
                    l2 = 'Finding' if raw_l2 == 'FindingTerm' else raw_l2
                    
                    rel = str(r['rel']) if r['rel'] is not None else 'CO-OCCURS'
                    
                    nodes.add((n1, l1))
                    nodes.add((n2, l2))
                    edges.append((n1, n2, rel))
            return list(nodes), edges
        except Exception as e:
            logger.error(f"Graph viz error: {e}")
            return [], []

    def run_cypher(self, cypher, params=None):
        with self.driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(cypher, params or {})]

    def get_schema_summary(self):
        queries = {
            "node_labels": "CALL db.labels() YIELD label RETURN collect(label) AS vals",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS vals",
            "patient_count": "MATCH (p:Patient) RETURN count(p) AS vals",
            "study_count": "MATCH (s:Study) RETURN count(s) AS vals",
            "finding_count": "MATCH (f:Finding) RETURN count(f) AS vals",
        }
        summary = {}
        for key, q in queries.items():
            try:
                with self.driver.session(database=self.database) as s:
                    rec = s.run(q).single()
                    summary[key] = rec["vals"] if rec else None
            except Exception as e: summary[key] = f"Error: {e}"
        return summary

class VectorReranker:
    def __init__(self, ollama):
        self.ollama = ollama
    @staticmethod
    def cosine(a, b):
        if not a or not b or len(a) != len(b): return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
        return 0.0 if not na or not nb else dot / (na * nb)
    def rerank(self, query, rows, topn=VECTOR_RERANK_TOPN):
        if not ENABLE_VECTOR_RERANK or not rows: return rows
        subset = rows[:topn]
        texts = tuple([query] + [f"{r.examname} | {r.modality} | {r.findingtext} | {' '.join(r.organs)}" for r in subset])
        vecs = self.ollama.embed_cached(texts, OLLAMA_EMBED_MODEL)
        qv = vecs[0] if vecs else []
        for row, vec in zip(subset, vecs[1:]):
            row.vector_score = self.cosine(qv, vec)
            row.final_score = round((0.78 * row.score) + (0.22 * row.vector_score * 10), 4)
        for row in rows[topn:]: row.final_score = row.score
        merged = subset + rows[topn:]; merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged

class GraphSearch:
    def __init__(self, gb, reranker):
        self.gb = gb
        self.reranker = reranker
    @staticmethod
    def score_result(result, parsed_query, raw_query):
        score, ft = 1.0, normalize_text(result.findingtext)
        organs, query_words = [normalize_text(o) for o in result.organs], [w for w in re.findall(r"[a-z0-9-]{3,}", normalize_text(raw_query)) if w not in STOPWORDS]
        if parsed_query.get("targetorgan"):
            oq = normalize_text(parsed_query["targetorgan"])
            score += 3.0 if any(oq in o for o in organs) else 1.4 if oq in ft else 0.0
        if parsed_query.get("finding") and normalize_text(parsed_query["finding"]) in ft: score += 2.4
        if parsed_query.get("modality") and parsed_query["modality"] in normalize_text(result.modality): score += 1.2
        if parsed_query.get("examname") and normalize_text(parsed_query["examname"]) in normalize_text(result.examname): score += 1.0
        if parsed_query.get("severity") and normalize_text(parsed_query["severity"]) == normalize_text(result.severity): score += 0.8
        if parsed_query.get("mindimensioncm") is not None and result.maxdimensioncm is not None:
            diff = abs(result.maxdimensioncm - float(parsed_query["mindimensioncm"]))
            score += max(0.0, 1.2 - min(diff, 1.2))
        keyword_hits = sum(1 for w in query_words if w in ft or any(w in o for o in organs) or w in normalize_text(result.clinicalinstruction))
        score += min(keyword_hits * 0.3, 1.8)
        if normalize_text(result.severity) in {"urgent", "critical", "malignant"}: score += 0.4
        return round(score, 4)

    @st.cache_data(ttl=120, show_spinner=False)
    def search(_self, parsed_query_json, raw_query, topk, advanced_filters_json, sort_by, include_text_search=True):
        parsed_query, advanced_filters = json.loads(parsed_query_json or "{}"), json.loads(advanced_filters_json or "{}")
        params = {"gender": None, "minage": None, "maxage": None, "modality": None, "examname": None, "targetorgan": None, "finding": None, "mindimensioncm": None, "maxdimensioncm": None, "severity": None, "topk": int(topk * 4)}
        for k in params.keys():
            if k == "topk": continue
            adv_val = advanced_filters.get(k)
            is_adv_default = ((k == "minage" and adv_val == 0) or (k == "maxage" and adv_val == 120) or (k == "maxdimensioncm" and adv_val == 50.0) or adv_val in (None, ""))
            if not is_adv_default: params[k] = adv_val
            elif k in parsed_query and parsed_query[k] not in (None, ""): params[k] = parsed_query[k]
        cypher = """
        MATCH (p:Patient)-[:HAS_STUDY]->(s:Study)
        OPTIONAL MATCH (s)-[:HAS_EXAM]->(e:Exam)
        OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality)
        OPTIONAL MATCH (s)-[:HAS_FINDING]->(f:Finding)
        OPTIONAL MATCH (f)-[:LOCATED_IN]->(o:Organ)
        OPTIONAL MATCH (f)-[:HAS_MEASUREMENT]->(meas:Measurement)
        WITH p, s, e, m, f, collect(DISTINCT toLower(coalesce(o.name_normalized,o.name,''))) AS organs_norm, collect(DISTINCT coalesce(o.name_normalized,o.name,'')) AS organs_display, max(meas.maxdimensioncm) AS maxdim
        WHERE ($gender IS NULL OR toLower(coalesce(p.gender,'')) STARTS WITH left(toLower($gender), 1))
          AND ($minage IS NULL OR p.age IS NULL OR toInteger(p.age) >= toInteger($minage))
          AND ($maxage IS NULL OR p.age IS NULL OR toInteger(p.age) <= toInteger($maxage))
          AND ($modality IS NULL OR toLower(coalesce(m.name,m.type,'')) CONTAINS toLower($modality) OR toLower($modality) CONTAINS toLower(coalesce(m.name,m.type,'')))
          AND ($examname IS NULL OR toLower(coalesce(e.name,e.type,'')) CONTAINS toLower($examname))
          AND ($severity IS NULL OR toLower(coalesce(s.severity,'')) CONTAINS toLower($severity))
          AND ($targetorgan IS NULL OR any(x IN organs_norm WHERE x CONTAINS toLower($targetorgan)) OR toLower(coalesce(f.text,s.resulttextplain,'')) CONTAINS toLower($targetorgan))
          AND ($finding IS NULL OR toLower(coalesce(f.text,s.resulttextplain,'')) CONTAINS toLower($finding))
          AND ($mindimensioncm IS NULL OR maxdim IS NULL OR maxdim >= toFloat($mindimensioncm) - 0.5)
          AND ($maxdimensioncm IS NULL OR maxdim IS NULL OR maxdim <= toFloat($maxdimensioncm) + 0.5)
        RETURN p.id AS hn, p.gender AS gender, p.age AS age, s.id AS studyid, s.studydate AS studydate, s.severity AS severity, s.clinicalinstruction AS clinicalinstruction, coalesce(m.name,m.type) AS modality, coalesce(e.name,e.type) AS examname, coalesce(f.text,s.resulttextplain) AS findingtext, organs_display AS organs, maxdim AS maxdimensioncm
        LIMIT $topk
        """
        rows = []
        try:
            with _self.gb.driver.session(database=_self.gb.database) as session:
                for r in session.run(cypher, params):
                    row = SearchResult(hn=r.get("hn") or "NA", gender=r.get("gender") or "?", age=r.get("age") if r.get("age") is not None else "?", studyid=r.get("studyid") or "", studydate=str(r.get("studydate") or ""), severity=r.get("severity") or "", clinicalinstruction=r.get("clinicalinstruction") or "", modality=r.get("modality") or "", examname=r.get("examname") or "", findingtext=r.get("findingtext") or "", organs=unique_preserve_order(r.get("organs") or []), maxdimensioncm=safe_float(r.get("maxdimensioncm")))
                    row.score = _self.score_result(row, parsed_query, raw_query); row.final_score = row.score; rows.append(row)
        except Exception as e: logger.error("Primary graph query failed: %s", e)
        if include_text_search and not rows and raw_query.strip():
            demographics = {"female", "woman", "male", "man", "age", "aged", "years", "old", "year", "with", "cm", "mm"}
            keywords = [w for w in re.findall(r"[a-z]{4,}", raw_query.lower()) if w not in STOPWORDS and w not in demographics][:6]
            if keywords:
                conds = [f"toLower(coalesce(f.text,s.resulttextplain,'')) CONTAINS '{k}'" for k in keywords]
                conds += ["($gender IS NULL OR toLower(coalesce(p.gender,'')) STARTS WITH left(toLower($gender), 1))", "($minage IS NULL OR p.age IS NULL OR toInteger(p.age) >= toInteger($minage))", "($maxage IS NULL OR p.age IS NULL OR toInteger(p.age) <= toInteger($maxage))", "($mindimensioncm IS NULL OR maxdim IS NULL OR maxdim >= toFloat($mindimensioncm) - 0.5)", "($maxdimensioncm IS NULL OR maxdim IS NULL OR maxdim <= toFloat($maxdimensioncm) + 0.5)"]
                fallback = f"MATCH (p:Patient)-[:HAS_STUDY]->(s:Study) OPTIONAL MATCH (s)-[:HAS_EXAM]->(e:Exam) OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality) OPTIONAL MATCH (s)-[:HAS_FINDING]->(f:Finding) OPTIONAL MATCH (f)-[:LOCATED_IN]->(o:Organ) OPTIONAL MATCH (f)-[:HAS_MEASUREMENT]->(meas:Measurement) WITH p,s,e,m,f,collect(DISTINCT coalesce(o.name,'')) AS organs,max(meas.maxdimensioncm) AS maxdim WHERE {' AND '.join(conds)} RETURN p.id AS hn, p.gender AS gender, p.age AS age, s.id AS studyid, s.studydate AS studydate, s.severity AS severity, s.clinicalinstruction AS clinicalinstruction, coalesce(m.name,m.type) AS modality, coalesce(e.name,e.type) AS examname, coalesce(f.text,s.resulttextplain) AS findingtext, organs AS organs, maxdim AS maxdimensioncm LIMIT $topk"
                try:
                    with _self.gb.driver.session(database=_self.gb.database) as session:
                        for r in session.run(fallback, params):
                            row = SearchResult(hn=r.get("hn") or "NA", gender=r.get("gender") or "?", age=r.get("age") or "?", studyid=r.get("studyid") or "", studydate=str(r.get("studydate") or ""), severity=r.get("severity") or "", clinicalinstruction=r.get("clinicalinstruction") or "", modality=r.get("modality") or "", examname=r.get("examname") or "", findingtext=r.get("findingtext") or "", organs=unique_preserve_order(r.get("organs") or []), maxdimensioncm=safe_float(r.get("maxdimensioncm")))
                            row.score = _self.score_result(row, parsed_query, raw_query); row.final_score = row.score; rows.append(row)
                except Exception as e: logger.warning("Fallback search failed: %s", e)
        rows = _self.reranker.rerank(raw_query, rows)
        if sort_by == "relevance": rows.sort(key=lambda x: x.final_score, reverse=True)
        elif sort_by == "age_asc": rows.sort(key=lambda x: safe_int(x.age, 0))
        elif sort_by == "age_desc": rows.sort(key=lambda x: safe_int(x.age, 0), reverse=True)
        elif sort_by == "size_desc": rows.sort(key=lambda x: safe_float(x.maxdimensioncm, 0.0) or 0.0, reverse=True)
        elif sort_by == "severity": rows.sort(key=lambda x: severity_rank(x.severity))
        dedup, seen = [], set()
        for row in rows:
            key = (row.hn, row.studyid, normalize_text(row.findingtext))
            if key not in seen: seen.add(key); dedup.append(row)
        return [r.as_dict() for r in dedup[:topk]]

class ClinicalContextBuilder:
    @staticmethod
    def build_context_string(results, max_results=MAX_CONTEXT_RESULTS):
        lines = []
        for i, r in enumerate(results[:max_results]):
            lines.append(f"{i+1}. HN:{r.get('hn','NA')} | {r.get('gender','?')}/{r.get('age','?')}y | {r.get('modality','?')} | {r.get('examname','?')} | Sev:{r.get('severity','NA')} | Findings:{truncate_text(r.get('findingtext',''), 300)}")
        return "\n".join(lines)
    @staticmethod
    def get_missing_organ_lines(results, query="", user_organs=None):
        modality, exam = next((r.get("modality", "") for r in results if r.get("modality")), ""), next((r.get("examname", "") for r in results if r.get("examname")), "")
        bucket = modality_to_study_bucket(modality, f"{query} {exam}")
        if not bucket or bucket not in STUDY_TEMPLATES: return []
        expected, present = STUDY_TEMPLATES[bucket]["organs"], {normalize_text(o) for r in results for o in (r.get("organs") or [])}
        for o in user_organs or []: present.add(normalize_text(o))
        return [f"{o.title()}: {STUDY_TEMPLATES[bucket]['default_text']}" for o in expected if normalize_text(o) not in present]

class ReportGenerator:
    def __init__(self, ollama): self.ollama = ollama
    def generate_report(self, query, search_results, report_type, system_prompt, user_organs=None, user_measurements=None, patient_info=None, report_template_key="General", extra_instructions=""):
        if not search_results: return "No evidence found."
        ctx, missing_lines = ClinicalContextBuilder.build_context_string(search_results), ClinicalContextBuilder.get_missing_organ_lines(search_results, query, user_organs)
        prompt = f"{system_prompt}\nGENERATE REPORT: Output format: {report_type}\nDate: {datetime.now().strftime('%d %B %Y')}\nEvidence:\n{ctx}\nAuto-lines:\n{chr(10).join(missing_lines)}\nGenerate now."
        return self.ollama.generate(prompt, OLLAMA_REPORT_MODEL, OLLAMA_TIMEOUT_SLOW, temperature=0.15)
    def answer_question(self, question, context, system_prompt, chat_history=None):
        prompt = f"{system_prompt}\nEvidence:\n{context}\nQuestion: {question}"
        return self.ollama.generate(prompt, OLLAMA_QA_MODEL, OLLAMA_TIMEOUT_SLOW, temperature=0.1)

# --- UI HELPERS ---
def init_state():
    defaults = {"search_results": [], "chat_history": [], "parsed_query": {}, "user_organs": [], "user_measurements": [], "last_report": "", "last_diffdx": "", "last_summary": "", "search_time_ms": 0, "raw_query_snapshot": "", "system_prompt_mode": "General Radiologist"}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
        
    if "selected_auto" not in st.session_state:
        st.session_state.selected_auto = []

def render_graph(nodes, edges):
    if not nodes: return
    dot = graphviz.Digraph(comment='Radiology Subgraph', engine='sfdp')
    dot.attr(bgcolor='transparent', overlap='false', splines='true')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Inter', fontsize='10', fontcolor='#e5edf6')
    
    colors = {'Organ': '#1e3a8a', 'FindingTerm': '#1e40af', 'Modality': '#1e1b4b', 'Exam': '#312e81', 'Finding': '#2563eb'}
    
    for n_name, n_label in nodes:
        safe_name = str(n_name) if n_name is not None else 'Unknown'
        color = colors.get(str(n_label), '#0f172a')
        dot.node(safe_name, safe_name, fillcolor=color, color='#26c6da')
    
    for n1, n2, rel in edges:
        s_n1 = str(n1) if n1 is not None else 'Unknown'
        s_n2 = str(n2) if n2 is not None else 'Unknown'
        s_rel = str(rel) if rel is not None else ''
        dot.edge(s_n1, s_n2, label=s_rel, color='#94a3b8', fontcolor='#94a3b8', fontsize='8')
    
    st.graphviz_chart(dot, use_container_width=True)

def build_sidebar(gb, ollama):
    with st.sidebar:
        st.markdown("### System")
        neo4j_ok, neo4j_msg = gb.verify(); ollama_ok, ollama_msg = ollama.is_ready(OLLAMA_MODEL)
        st.markdown(f'<span class="status-badge {"badge-ok" if neo4j_ok else "badge-err"}">* Neo4j</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge {"badge-ok" if ollama_ok else "badge-err"}">* Ollama</span>', unsafe_allow_html=True)
        st.divider()
        st.markdown("### Search filters")
        fopts = gb.get_filter_options("{}")
        adv_gender = st.selectbox("Gender", ["", "Male", "Female"])
        adv_min_age, adv_max_age = st.slider("Age range", 0, 120, (0, 120))
        adv_modality = st.selectbox("Modality", [""] + fopts.get("modalities", []))
        adv_exam = st.selectbox("Exam", [""] + fopts.get("exams", []))
        adv_organ = st.selectbox("Target organ", [""] + KNOWN_ORGANS)
        adv_sev = st.selectbox("Severity", [""] + fopts.get("severity", []))
        adv_min_dim = st.number_input("Min size (cm)", 0.0, 50.0, 0.0)
        adv_max_dim = st.number_input("Max size (cm)", 0.0, 50.0, 50.0)
        topk = st.slider("Top K", 5, 100, DEFAULT_TOPK)
        sort_by = st.selectbox("Sort by", ["relevance", "severity", "size_desc", "age_desc", "age_asc"])
        persona = st.selectbox("AI persona", list(QA_SYSTEM_PROMPTS.keys()))
    return {"gender": adv_gender, "minage": adv_min_age, "maxage": adv_max_age, "modality": adv_modality, "examname": adv_exam, "targetorgan": adv_organ, "severity": adv_sev, "mindimensioncm": adv_min_dim if adv_min_dim > 0 else None, "maxdimensioncm": adv_max_dim if adv_max_dim < 50 else None, "topk": topk, "sort_by": sort_by, "persona": persona}

def run_streamlit_app():
    st.set_page_config(page_title=APP_TITLE, page_icon="xray", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    gb, ollama = GraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DB), OllamaClient(OLLAMA_URL)
    qp, gs, rg = QueryParserLLM(ollama), GraphSearch(gb, VectorReranker(ollama)), ReportGenerator(ollama)
    
    st.markdown(f'<div class="main-header"><h1>{APP_TITLE}</h1><p>v{APP_VERSION} | Co-occurrence Graph Search & Synthesis</p></div>', unsafe_allow_html=True)
    sidebar = build_sidebar(gb, ollama)
    
    tabs = st.tabs(["Clinical Records", "AI QA", "Report Generator", "Cypher", "Status"])
    
    with tabs[0]:
        st.markdown("### Database Search")
        
        autoterms = gb.get_dynamic_autocomplete_terms(tuple(st.session_state.selected_auto))
        
        st.multiselect(
            "Select from database terms (e.g., Finding: nodule, Organ: liver)", 
            options=autoterms, 
            key="selected_auto"
        )
        
        if st.session_state.selected_auto:
            with st.container():
                st.markdown("#### Clinical Co-occurrence Graph")
                nodes, edges = gb.get_subgraph_for_terms(st.session_state.selected_auto)
                if nodes:
                    c_graph, c_suggest = st.columns([3, 1])
                    with c_graph:
                        render_graph(nodes, edges)
                    with c_suggest:
                        st.caption("Frequently Co-occurring Nodes (Click to add)")
                        
                        selected_names = [t.split(": ", 1)[1] if ": " in t else t for t in st.session_state.selected_auto]
                        suggested_terms = []
                        for n1, n2, rel in edges:
                            lbl2 = next((lbl for nn, lbl in nodes if nn == n2), 'Unknown')
                            fmt_term = f"{lbl2}: {n2}"
                            if n2 not in selected_names and fmt_term not in suggested_terms and lbl2 != 'Unknown':
                                suggested_terms.append(fmt_term)
                                
                        for term in suggested_terms[:12]:
                            if st.button(term, key=f"add_{term}"):
                                st.session_state.selected_auto.append(term)
                                st.rerun()
                else:
                    st.caption("No internal relationships found for the selected terms.")
        
        colq, colbtn = st.columns([5, 1])
        with colq: free_text = st.text_input("Natural language query", placeholder="e.g. 1.5 cm thyroid nodule on US")
        with colbtn: st.write(""); search_btn = st.button("Search", type="primary", use_container_width=True)
        
        if search_btn:
            query = f"{free_text.strip()} {' '.join(st.session_state.selected_auto)}".strip()
            if query:
                t0 = time.time()
                parsed = qp.parse_query(query)
                st.session_state.user_organs = extract_organs_from_text(query)
                st.session_state.user_measurements = extract_measurements_from_text(query)
                advanced = {k: sidebar[k] for k in ["gender", "minage", "maxage", "modality", "examname", "targetorgan", "severity", "mindimensioncm", "maxdimensioncm"]}
                st.session_state.search_results = gs.search(json.dumps(parsed), query, int(sidebar["topk"]), json.dumps(advanced), sidebar["sort_by"], True)
                st.session_state.parsed_query, st.session_state.search_time_ms, st.session_state.raw_query_snapshot = parsed, int((time.time()-t0)*1000), query

        if st.session_state.search_results:
            df = pd.DataFrame(st.session_state.search_results)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("### Clinical QA")
        if not st.session_state.search_results: st.info("Search first.")
        else:
            context = ClinicalContextBuilder.build_context_string(st.session_state.search_results)
            user_q = st.text_input("Ask about results")
            if st.button("Ask AI") and user_q:
                ans = rg.answer_question(user_q, context, QA_SYSTEM_PROMPTS[sidebar["persona"]])
                st.session_state.chat_history.append({"role":"user", "content":user_q})
                st.session_state.chat_history.append({"role":"assistant", "content":ans})
            for msg in reversed(st.session_state.chat_history):
                st.markdown(f"**{msg['role'].upper()}**: {msg['content']}")

    with tabs[2]:
        st.markdown("### Report Generation")
        if st.button("Generate Pro Report") and st.session_state.search_results:
            report = rg.generate_report(st.session_state.raw_query_snapshot, st.session_state.search_results, "Full Report", QA_SYSTEM_PROMPTS[sidebar["persona"]])
            st.markdown(f'<div class="report-output">{report}</div>', unsafe_allow_html=True)

    with tabs[3]:
        cypher = st.text_area("Cypher", "MATCH (n) RETURN n LIMIT 10")
        if st.button("Run"): st.write(gb.run_cypher(cypher))

    with tabs[4]:
        st.write(gb.get_schema_summary())

if __name__ == "__main__":
    run_streamlit_app()



# CREATE INDEX organ_name_idx IF NOT EXISTS FOR (n:Organ) ON (n.name);
# CREATE INDEX findingterm_name_idx IF NOT EXISTS FOR (n:FindingTerm) ON (n.name);
# CREATE INDEX exam_name_idx IF NOT EXISTS FOR (n:Exam) ON (n.name);
# CREATE INDEX modality_name_idx IF NOT EXISTS FOR (n:Modality) ON (n.name);
# CREATE INDEX modality_type_idx IF NOT EXISTS FOR (n:Modality) ON (n.type);
# CREATE INDEX patient_gender_idx IF NOT EXISTS FOR (p:Patient) ON (p.gender);
# CREATE INDEX patient_age_idx IF NOT EXISTS FOR (p:Patient) ON (p.age);
# CREATE INDEX study_severity_idx IF NOT EXISTS FOR (s:Study) ON (s.severity);
# CREATE INDEX measurement_dim_idx IF NOT EXISTS FOR (m:Measurement) ON (m.maxdimensioncm);