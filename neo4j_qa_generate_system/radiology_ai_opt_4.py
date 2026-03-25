import os
import re
import sys
import json
import logging
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple, Iterable
import requests
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

# =========================================================
# CONFIG - PRODUCTION READY FOR 60M+ NEO4J (NO EMBEDDINGS)
# =========================================================
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("radiology_ai_workbench")

APP_TITLE = "🏥 AI Radiology Workbench - •Neo4j •Offline Ollama"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "12345678")

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "15"))

HEADER_ALIASES = {
    "History": [r"HISTORY", r"CLINICAL INFORMATION", r"CLINICAL INSTRUCTION"],
    "Technique": [r"TECHNIQUE", r"TECHNIQUES"],
    "Comparison": [r"COMPARISON", r"COMPARED WITH DATE"],
    "Findings": [
        r"[\w\s-]*FINDINGS?", r"US-GUIDED BIOPSY", r"LUNGS?(?: AND PLEURA)?",
        r"PLEURA AND DIAPHRAGMS", r"HEART AND MEDIASTINUM", r"BONY THORAX",
        r"BONES AND JOINTS", r"ALIGNMENT", r"SOFT TISSUES"
    ],
    "Impression": [r"IMPRESSION", r"IMP"],
    "Recommendation": [r"RECOMMENDATION", r"PLAN"]
}

KNOWN_MODALITIES = {
    "us": ["us", "ultrasound", "sonography"],
    "ct": ["ct", "computed tomography"],
    "mri": ["mri", "mr", "magnetic resonance"],
    "xray": ["xray", "x-ray", "xr", "radiograph"]
}

KNOWN_ORGANS = [
    "thyroid", "right thyroid lobe", "left thyroid lobe", "isthmus", "cervical lymph nodes",
    "lymph nodes", "liver", "gallbladder", "pancreas", "spleen", "kidneys", "kidney",
    "adrenals", "adrenal", "bladder", "bowel loops", "vasculature", "lungs", "lung",
    "pleura", "heart", "mediastinum", "hilar structures"
]

KNOWN_FINDINGS = [
    "nodule", "cyst", "microcalcifications", "enlarged", "lesion", "mass", 
    "calcification", "hypoechoic", "heterogeneous", "normal", "unremarkable"
]

STOPWORDS = {"show", "find", "for", "with", "and", "the", "a", "an", "of", "in", "on", "patient"}

STUDY_TEMPLATES = {
    "abdomen": {"organs": ["liver", "gallbladder", "spleen", "pancreas", "kidneys", "adrenals", "bladder", "bowel loops", "vasculature"], "default_text": "Unremarkable."},
    "chest": {"organs": ["lungs", "pleura", "heart", "mediastinum", "hilar structures"], "default_text": "Unremarkable."},
    "thyroid": {"organs": ["right thyroid lobe", "left thyroid lobe", "isthmus", "cervical lymph nodes"], "default_text": "No focal abnormality identified."},
}


# =========================================================
# HELPERS
# =========================================================
def normalize_text(value: Any) -> str:
    if value is None: return ""
    text = str(value).strip().lower()
    return re.sub(r"[\s\r\n\t]+", " ", text)

def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen, result = set(), []
    for value in values:
        if value is None: continue
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(str(value))
    return result

def safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try: return default if value in (None, "") else int(float(value))
    except Exception: return default

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try: return default if value in (None, "") else float(value)
    except Exception: return default

def parse_size_to_cm(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)\b", str(text), re.IGNORECASE)
    if not m: return None
    value, unit = float(m.group(1)), m.group(2).lower()
    return value / 10.0 if unit == "mm" else value

def truncate_text(text: str, max_len: int = 800) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

def modality_to_study_bucket(modality_text: str, exam_text: str) -> Optional[str]:
    combined = normalize_text(f"{modality_text} {exam_text}")
    if any(k in combined for k in ["abdomen", "hepato", "liver", "gallbladder", "pancrea", "renal"]): return "abdomen"
    if any(k in combined for k in ["chest", "lung", "pleura", "thorax", "mediast"]): return "chest"
    if "thyroid" in combined or "neck" in combined: return "thyroid"
    return None

# =========================================================
# ONTOLOGY MAPPER
# =========================================================
class OntologyMapper:
    def __init__(self):
        self.anatomy_map = {o: ("RadLex", f"RID{i}") for i, o in enumerate(KNOWN_ORGANS)}
        self.findings_map = {f: ("SNOMED", f"SNO{i}") for i, f in enumerate(KNOWN_FINDINGS)}

    def map_anatomy(self, term: str) -> Tuple[str, str]:
        t = normalize_text(term)
        return self.anatomy_map.get(t, ("Unknown", "UNK"))

    def map_finding(self, term: str) -> Tuple[str, str]:
        t = normalize_text(term)
        return self.findings_map.get(t, ("Unknown", "UNK"))

# =========================================================
# REPORT PARSER & ENTITY EXTRACTION
# =========================================================
class ReportParser:
    def __init__(self):
        self.aliases = HEADER_ALIASES

    def parse_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        if not text: return sections
        header_patterns = [f"(?P<{header}_{idx}>{pattern}\\s*:)" for header, patterns in self.aliases.items() for idx, pattern in enumerate(patterns)]
        combined_pattern = "|".join(header_patterns)
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE | re.MULTILINE))

        if not matches:
            sections["unstructured"] = text.strip()
            return sections

        if matches[0].start() > 0: sections["Header"] = text[:matches[0].start()].strip()
        for i, match in enumerate(matches):
            header_name = match.lastgroup.split("_")[0]
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_idx:end_idx].strip()
            if content: sections[header_name] = (sections.get(header_name, "") + "\n" + content).strip()
        return sections

class EntityExtractor:
    def __init__(self):
        self.ontology_mapper = OntologyMapper()
        self.meas_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[xX\*]\s*(\d+(?:\.\d+)?))?\s*(cm|mm)", re.IGNORECASE)
        self.anatomies = sorted(KNOWN_ORGANS, key=len, reverse=True)
        self.findings = sorted(KNOWN_FINDINGS, key=len, reverse=True)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip(): return []
        text_norm = normalize_text(text)
        measurements, found_anatomies, found_findings = [], [], []

        for match in self.meas_pattern.finditer(text):
            dims = [float(d) for d in match.groups()[:-1] if d is not None]
            unit = match.groups()[-1].lower()
            dims_cm = [round(d / 10.0 if unit == "mm" else d, 4) for d in dims]
            measurements.append({
                "value": " x ".join(map(str, dims)) + f" {unit}",
                "dimensions_cm": dims_cm,
                "max_dimension_cm": max(dims_cm) if dims_cm else None,
                "raw_text": match.group().strip()
            })

        for anat in self.anatomies:
            if normalize_text(anat) in text_norm:
                onto_db, onto_id = self.ontology_mapper.map_anatomy(anat)
                found_anatomies.append({"name": anat, "name_normalized": normalize_text(anat), "ontology": onto_db, "concept_id": onto_id})

        for term in self.findings:
            if normalize_text(term) in text_norm:
                onto_db, onto_id = self.ontology_mapper.map_finding(term)
                found_findings.append({"name": term, "name_normalized": normalize_text(term), "ontology": onto_db, "concept_id": onto_id})

        if found_anatomies or found_findings or measurements:
            return [{
                "context_block": text.strip(),
                "context_block_normalized": text_norm,
                "organs": [dict(t) for t in {tuple(sorted(x.items())) for x in found_anatomies}],
                "finding_terms": [dict(t) for t in {tuple(sorted(x.items())) for x in found_findings}],
                "measurements": measurements
            }]
        return []

    def process_sections(self, sections: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        section_entities = {}
        for section_name, content in sections.items():
            if section_name.lower() in ["findings", "impression", "unstructured"]:
                lines = [line.strip(" -•\t") for line in content.split("\n") if line.strip()]
                entities = [ent for line in lines for ent in self.extract_entities(line)]
                if not entities and content.strip():
                    entities = [{"context_block": content.strip(), "context_block_normalized": normalize_text(content), "organs": [], "finding_terms": [], "measurements": []}]
                if entities: section_entities[section_name] = entities
        return section_entities

# =========================================================
# QUERY PARSING (LLM for complex logic, multiple ranges)
# =========================================================
class QueryParserLLM:
    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.url = url
        self.model = model

    def parse_query(self, query: str) -> dict:
        if not query.strip(): return {}
        
        prompt = f"""
You are a medical data extraction API. Extract search parameters from the user's clinical query.
Return ONLY valid JSON. Omit keys if data is absent. Convert mm to cm.
If multiple age ranges exist (e.g. 10 to 30, 50 to 60), pick the absolute min and max.
If "Explicitly selected parameters" are provided in the query, use them strictly for their respective keys (e.g., Organ goes to target_organ, Finding goes to finding, History goes to keywords).
Allowed keys: gender, min_age, max_age, modality, target_organ, finding, min_dimension_cm, max_dimension_cm, keywords (array of strings)

Query: "{query}"
JSON:
""".strip()

        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        try:
            response = requests.post(f"{self.url}/api/generate", json=payload, timeout=10)
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            logger.warning(f"LLM parse failed: {e}")
            return {}

# =========================================================
# GRAPH BUILDER (Cascading Options & Categorized Autocomplete)
# =========================================================
class GraphBuilder:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self): self.driver.close()

    def verify(self) -> Tuple[bool, str]:
        try:
            with self.driver.session() as session:
                rec = session.run("RETURN 1 AS ok").single()
                return (True, "Neo4j connected") if rec and rec["ok"] == 1 else (False, "Failed")
        except Exception as e: return False, str(e)

    def get_filter_options(self, filters: dict = None, limit: int = 100) -> Dict[str, List[str]]:
        f = filters or {}
        modality = normalize_text(f.get("modality", ""))
        severity = normalize_text(f.get("severity", ""))

        def fetch_list(cypher: str, params: dict) -> List[str]:
            try:
                with self.driver.session() as session:
                    rec = session.run(cypher, **params).single()
                    return [x for x in (rec["vals"] if rec else []) if x]
            except Exception as e: 
                logger.warning(f"Filter fetch failed: {e}")
                return []

        # Fast global index scan if no filters are active
        if not modality and not severity:
            queries = {
                "exams": "MATCH (e:Exam) RETURN collect(DISTINCT e.name)[0..$limit] AS vals",
                "modalities": "MATCH (m:Modality) RETURN collect(DISTINCT m.name)[0..$limit] AS vals",
                "severity": "MATCH (s:Study) WHERE s.severity IS NOT NULL RETURN collect(DISTINCT s.severity)[0..$limit] AS vals"
            }
            return {k: unique_preserve_order(fetch_list(q, {"limit": limit})) for k, q in queries.items()}

        # CASCADING search based on relationships
        cascade_queries = {
            "exams": """
                MATCH (s:Study)-[:HAS_EXAM]->(e:Exam)
                OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality)
                WHERE ($modality = "" OR toLower(m.name) CONTAINS $modality)
                  AND ($severity = "" OR toLower(s.severity) CONTAINS $severity)
                RETURN collect(DISTINCT e.name)[0..$limit] AS vals
            """,
            "modalities": """
                MATCH (s:Study)-[:HAS_MODALITY]->(m:Modality)
                WHERE ($severity = "" OR toLower(s.severity) CONTAINS $severity)
                RETURN collect(DISTINCT m.name)[0..$limit] AS vals
            """,
            "severity": """
                MATCH (s:Study)-[:HAS_MODALITY]->(m:Modality)
                WHERE ($modality = "" OR toLower(m.name) CONTAINS $modality)
                  AND s.severity IS NOT NULL
                RETURN collect(DISTINCT s.severity)[0..$limit] AS vals
            """
        }
        
        params = {"limit": limit, "modality": modality, "severity": severity}
        return {k: unique_preserve_order(fetch_list(q, params)) for k, q in cascade_queries.items()}

    def get_autocomplete_terms(self, limit: int = 500) -> List[str]:
        # Prepend category so the LLM explicitly knows what field to put it in
        query = """
        CALL () {
            MATCH (ft:FindingTerm) RETURN DISTINCT 'Finding: ' + ft.name AS term LIMIT $limit
            UNION
            MATCH (o:Organ) RETURN DISTINCT 'Organ: ' + o.name AS term LIMIT $limit
            UNION
            MATCH (s:Study) WHERE s.clinical_instruction IS NOT NULL RETURN DISTINCT 'History: ' + s.clinical_instruction AS term LIMIT $limit
        }
        RETURN term
        """
        try:
            with self.driver.session() as session:
                rows = session.run(query, limit=limit)
                return sorted([r["term"] for r in rows if r["term"]])
        except Exception as e:
            logger.warning(f"Failed to fetch autocomplete terms: {e}")
            return []

# =========================================================
# FAST GRAPH SEARCH (No Embeddings, Null-Safe)
# =========================================================
class GraphSearch:
    def __init__(self, graph_builder: GraphBuilder):
        self.gb = graph_builder

    def search(self, parsed_query: dict, raw_query: str, top_k: int = DEFAULT_TOP_K, advanced_filters: dict = None) -> List[Dict[str, Any]]:
        # Initialize ALL expected Neo4j parameters to prevent ParameterMissing errors
        params = {
            "gender": None,
            "min_age": None,
            "max_age": None,
            "modality": None,
            "exam_name": None,
            "target_organ": None,
            "min_dimension_cm": None,
            "max_dimension_cm": None,
            "severity": None,
            "top_k": top_k
        }

        # Overlay parsed LLM query parameters
        if parsed_query:
            for k in params.keys():
                if k in parsed_query and parsed_query[k] not in (None, "", []):
                    params[k] = parsed_query[k]

        # Overlay Manual UI Advanced Filters (Cascading selections)
        if advanced_filters:
            for k in params.keys():
                if k in advanced_filters and advanced_filters[k] not in (None, "", []):
                    params[k] = advanced_filters[k]

        # Final safety check: convert lingering empty strings to None
        for k, v in params.items():
            if isinstance(v, str) and not v.strip():
                params[k] = None

        # Bulletproof, Null-Safe Cypher for 60M+
        query = """
        MATCH (p:Patient)-[:HAS_STUDY]->(s:Study)
        OPTIONAL MATCH (s)-[:HAS_EXAM]->(e:Exam)
        OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality)
        OPTIONAL MATCH (s)-[:HAS_FINDING]->(f:Finding)
        OPTIONAL MATCH (f)-[:LOCATED_IN]->(o:Organ)
        OPTIONAL MATCH (f)-[:HAS_MEASUREMENT]->(meas:Measurement)
        
        WITH p, s, e, m, f,
             collect(DISTINCT toLower(coalesce(o.name_normalized, ""))) AS organs_norm,
             max(meas.max_dimension_cm) AS max_dimension_cm

        WHERE 
            ($gender IS NULL OR toLower(coalesce(p.gender, "")) = toLower($gender))
            AND ($min_age IS NULL OR toInteger(coalesce(p.age, 0)) >= toInteger($min_age))
            AND ($max_age IS NULL OR toInteger(coalesce(p.age, 0)) <= toInteger($max_age))
            AND ($modality IS NULL OR toLower(coalesce(m.name, m.type, "")) CONTAINS toLower($modality))
            AND ($exam_name IS NULL OR toLower(coalesce(e.name, e.type, "")) CONTAINS toLower($exam_name))
            AND ($severity IS NULL OR toLower(coalesce(s.severity, s.severity_normalized, "")) CONTAINS toLower($severity))
            AND (
                $target_organ IS NULL 
                OR any(x IN organs_norm WHERE x CONTAINS toLower($target_organ))
                OR toLower(coalesce(f.text, s.result_text_plain, "")) CONTAINS toLower($target_organ)
            )
            AND ($min_dimension_cm IS NULL OR max_dimension_cm >= toFloat($min_dimension_cm))
            AND ($max_dimension_cm IS NULL OR max_dimension_cm <= toFloat($max_dimension_cm))
            
        RETURN 
            p.id AS hn, p.gender AS gender, p.age AS age, s.id AS study_id, 
            s.severity AS severity, s.clinical_instruction AS clinical_instruction,
            coalesce(m.name, m.type) AS modality, e.name AS exam_name,
            coalesce(f.text, s.result_text_plain) AS finding_text, 
            organs_norm AS organs, max_dimension_cm
        LIMIT $top_k
        """
        
        try:
            with self.gb.driver.session() as session:
                rows = session.run(query, **params)
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# =========================================================
# CLINICAL CONTEXT & REPORT GENERATOR
# =========================================================
class ClinicalContextBuilder:
    @staticmethod
    def generate_missing_organ_lines(results: List[Dict[str, Any]], query: str = "", user_mentioned_organs: List[str] = None) -> List[str]:
        bucket = modality_to_study_bucket(next((r.get("modality", "") for r in results if r.get("modality")), ""), query)
        if not bucket or bucket not in STUDY_TEMPLATES: return []
        
        expected = STUDY_TEMPLATES[bucket]["organs"]
        default_text = STUDY_TEMPLATES[bucket]["default_text"]
        present = set(normalize_text(o) for r in results for o in (r.get("organs") or []))
        
        if user_mentioned_organs:
            for o in user_mentioned_organs: present.add(normalize_text(o))
                
        missing = [org for org in expected if normalize_text(org) not in present]
        return [f"{org.title()}: {default_text}" for org in missing]

class ReportGenerator:
    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.url = url
        self.model = model

    def is_ready(self) -> Tuple[bool, str]:
        try:
            requests.post(f"{self.url}/api/generate", json={"model": self.model, "prompt": "hi", "stream": False}, timeout=5)
            return True, "Ollama connected"
        except Exception as e: return False, str(e)

    def generate_report(self, query: str, search_results: List[Dict[str, Any]], report_type: str, system_prompt: str, user_mentioned_organs: List[str] = None, user_mentioned_measurements: List[str] = None) -> str:
        if not search_results: return "No evidence found. Adjust filters."
        
        context_payload = json.dumps([{k: str(v)[:200] for k, v in r.items() if v} for r in search_results[:MAX_CONTEXT_RESULTS]])
        missing_lines = ClinicalContextBuilder.generate_missing_organ_lines(search_results, query, user_mentioned_organs)
        
        # Build strict explicit instructions for the LLM
        extra_instruction = ""
        if user_mentioned_organs or user_mentioned_measurements:
            orgs_text = ", ".join(user_mentioned_organs) if user_mentioned_organs else "None specified"
            meas_text = ", ".join(user_mentioned_measurements) if user_mentioned_measurements else "None specified"
            extra_instruction = f"\nCRITICAL: The user explicitly requested these parameters in their query -> Organs: [{orgs_text}] | Measurements: [{meas_text}]. You MUST force these exact organs and measurements into the report correctly."
        
        output_hint = {
            "Full Report": "Return Findings, Impression, and Recommendations.",
            "Only Findings": "Return only the Findings section.",
            "Only Impression": "Return only Impression section in numbered form.",
            "Recommendations": "Return only Recommendations."
        }.get(report_type, "Return a complete professional report.")
        
        prompt = f"""
{system_prompt}
You are generating a production radiology report.
Rules:
- Include EVERY explicitly requested organ and measurement.
- Add auto-generated missing organ statements EXACTLY as provided below.
{extra_instruction}

Requested format:
{output_hint}

Evidence payload:
{context_payload}

Auto-Added Unremarkable Organs to include:
{chr(10).join(missing_lines) if missing_lines else "None"}
"""
        try:
            resp = requests.post(f"{self.url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False}, timeout=120)
            return resp.json().get("response", "").strip()
        except Exception:
            return "LLM generation failed. Check Ollama."

    def answer_question(self, question: str, context: str) -> str:
        prompt = f"Answer concisely based on this clinical evidence ONLY:\n{context}\n\nQuestion: {question}"
        try:
            return requests.post(f"{self.url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False}, timeout=60).json().get("response", "")
        except Exception: return "Failed to reach Ollama."

# =========================================================
# STREAMLIT UI
# =========================================================
def run_streamlit_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🏥")
    st.title(APP_TITLE)
    st.caption("• Neo4j • Pure Graph Search (No Embeddings) • Cascading Filters • Database Auto-Suggest")

    @st.cache_resource
    def init_services():
        gb = GraphBuilder()
        return gb, QueryParserLLM(), GraphSearch(gb), ReportGenerator()

    gb, qp, gs, rg = init_services()

    # Initialize State variables
    for state in ["search_results", "chat_history", "parsed_query", "user_mentioned_organs", "user_mentioned_measurements", "last_report"]:
        if state not in st.session_state: 
            st.session_state[state] = [] if state not in ["parsed_query", "last_report"] else {} if state == "parsed_query" else ""

    # CASCADE FILTERS (SIDEBAR)
    with st.sidebar:
        st.header("🔬 Live Cascade Filters")
        st.caption("Options shrink based on your selections")
        
        # Read CURRENT state of dropdowns to cascade queries
        current_filters = {
            "modality": st.session_state.get("adv_mod", ""),
            "severity": st.session_state.get("adv_sev", "")
        }
        opts = gb.get_filter_options(filters=current_filters, limit=50)

        # Rendering input fields (st.session_state will auto-update based on keys)
        adv_gender = st.selectbox("Gender", ["", "Male", "Female"], key="adv_gender")
        
        col1, col2 = st.columns(2)
        with col1:
            adv_min_age = st.number_input("Min Age", 0, 120, 0, key="adv_min_age")
        with col2:
            adv_max_age = st.number_input("Max Age", 0, 120, 120, key="adv_max_age")
            
        adv_mod = st.selectbox("Modality", [""] + opts.get("modalities", []), key="adv_mod")
        adv_exam = st.selectbox("Exam Type", [""] + opts.get("exams", []), key="adv_exam")
        adv_organ = st.selectbox("Organ", [""] + sorted(KNOWN_ORGANS), key="adv_organ")
        adv_sev = st.selectbox("Severity", [""] + opts.get("severity", []), key="adv_sev")
        
        if st.button("Reset Filters", use_container_width=True):
            for key in ["adv_gender", "adv_min_age", "adv_max_age", "adv_mod", "adv_exam", "adv_organ", "adv_sev"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # MAIN AREA
    st.markdown("### 🔍 Smart Clinical Search")
    
    # Cache autocomplete terms so UI doesn't lag
    @st.cache_data(ttl=3600)
    def load_auto_terms():
        return gb.get_autocomplete_terms(limit=1000)
    
    auto_terms = load_auto_terms()

    # The Autocomplete Multi-select (Injects Neo4j DB terms with explicit Categories)
    selected_auto_terms = st.multiselect(
        "⚡ Type to auto-complete clinical terms from the database:", 
        options=auto_terms,
        placeholder="e.g., Finding: microcalcifications, Organ: left thyroid lobe..."
    )

    col_query, col_k = st.columns([5, 1])
    with col_query:
        free_text_query = st.text_input(
            "💬 Natural Language Query (or ask for a report)", 
            placeholder="female patient age 10 to 30 suffering from fever with 1.5 cm cyst..."
        )
    with col_k:
        top_k = st.number_input("Top K", 1, 100, DEFAULT_TOP_K)

    # Combine natural language query with explicitly categorized DB terms for the LLM
    explicit_params = ", ".join(selected_auto_terms)
    if explicit_params:
        query = f"{free_text_query} (Explicitly selected parameters: {explicit_params})".strip()
    else:
        query = free_text_query.strip()

    sys_prompt = st.text_area("System Prompt", "You are a board-certified radiologist.", height=68)
    report_type = st.radio("Report Type", ["Full Report", "Only Findings", "Only Impression", "Recommendations"], horizontal=True)

    if st.button("🚀 Search & Process", type="primary", use_container_width=True):
        if not query and not any([st.session_state.get(k) for k in ["adv_gender", "adv_min_age", "adv_mod", "adv_exam", "adv_organ", "adv_sev"]]):
            st.warning("Please enter a query or select a filter before searching.")
        else:
            with st.spinner("Searching 60M+ graph..."):
                parsed = qp.parse_query(query)
                
                # --- DEBUG BLOCK FOR LLM PARSING ---
                with st.expander("🛠️ Debug: What the LLM understood (Click to view)", expanded=False):
                    st.json(parsed)
                
                # 1. Extract explicit organs from the user's free text for the report
                st.session_state.user_mentioned_organs = [o for o in KNOWN_ORGANS if re.search(rf"\b{o}\b", query, re.IGNORECASE)]
                
                # 2. Extract explicit measurements (e.g., 1.5 cm, 2x3 mm) from the text for the report
                meas_pattern = r"(\d+(?:\.\d+)?\s*(?:[xX\*]\s*\d+(?:\.\d+)?\s*)*(?:cm|mm))"
                extracted_measurements = re.findall(meas_pattern, query, re.IGNORECASE)
                st.session_state.user_mentioned_measurements = [m.strip() for m in extracted_measurements]

                advanced = {
                    "gender": adv_gender, "min_age": adv_min_age, "max_age": adv_max_age, 
                    "modality": adv_mod, "exam_name": adv_exam, "target_organ": adv_organ, "severity": adv_sev
                }
                
                st.session_state.search_results = gs.search(parsed, query, top_k, advanced)
                if st.session_state.search_results:
                    st.success(f"✅ Found {len(st.session_state.search_results)} records.")
                else:
                    st.warning("No matches found. Check the Debug expander above to see if the LLM parsed a constraint too strictly.")

    tab_data, tab_chat, tab_report, tab_system = st.tabs([
        "🗂️ Fetched Records", "💬 AI Chat", "📋 Report", "🔧 System Status"
    ])

    with tab_data:
        if st.session_state.search_results:
            st.subheader(f"Retrieved Studies ({len(st.session_state.search_results)})")
            
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(st.session_state.search_results)
            
            # Reorder columns logically
            cols = df.columns.tolist()
            preferred_order = ['hn', 'age', 'gender', 'modality', 'exam_name', 'severity', 'finding_text', 'max_dimension_cm', 'organs']
            ordered_cols = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
            
            st.dataframe(
                df[ordered_cols], 
                use_container_width=True,
                hide_index=True,
                height=500
            )
        else:
            st.info("No records fetched yet. Run a search to see the data here!")

    with tab_chat:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if user_msg := st.chat_input("Ask about the retrieved data..."):
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            st.chat_message("user").markdown(user_msg)
            with st.spinner("Thinking..."):
                context = json.dumps(st.session_state.search_results[:5])
                ans = rg.answer_question(user_msg, context)
                st.chat_message("assistant").markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

    with tab_report:
        if st.button("✨ Generate Professional Report", use_container_width=True):
            with st.spinner("Generating..."):
                rep = rg.generate_report(
                    query=query, 
                    search_results=st.session_state.search_results, 
                    report_type=report_type, 
                    system_prompt=sys_prompt, 
                    user_mentioned_organs=st.session_state.user_mentioned_organs,
                    user_mentioned_measurements=st.session_state.user_mentioned_measurements
                )
                st.session_state.last_report = rep
        if st.session_state.last_report:
            st.text_area("Report", st.session_state.last_report, height=400)

    with tab_system:
        neo_ok, neo_msg = gb.verify()
        llm_ok, llm_msg = rg.is_ready()
        st.metric("Neo4j", "✅ OK" if neo_ok else "❌ Down", neo_msg)
        st.metric("Ollama", "✅ OK" if llm_ok else "❌ Down", llm_msg)

if __name__ == "__main__":
    if "streamlit" in sys.modules: run_streamlit_app()
    else: os.system("streamlit run " + sys.argv[0])