import os
import re
import sys
import json
import time
import math
import queue
import logging
import argparse
import warnings
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Tuple, Iterable

import requests
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


# =========================================================
# CONFIG
# =========================================================
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("radiology_workbench")

APP_TITLE = "🏥 Production Radiology QA, Search & Report System"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "12345678")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "abhinand/MedEmbed-large-v0.1")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
VECTOR_ENABLED = os.getenv("VECTOR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}
MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", "12"))


HEADER_ALIASES = {
    "History": [r"HISTORY", r"CLINICAL INFORMATION", r"CLINICAL INSTRUCTION"],
    "Technique": [r"TECHNIQUE", r"TECHNIQUES"],
    "Comparison": [r"COMPARISON", r"COMPARED WITH DATE"],
    "Findings": [
        r"[\w\s-]*FINDINGS?",
        r"US-GUIDED BIOPSY",
        r"LUNGS?(?: AND PLEURA)?",
        r"PLEURA AND DIAPHRAGMS",
        r"HEART AND MEDIASTINUM",
        r"BONY THORAX",
        r"BONES AND JOINTS",
        r"ALIGNMENT",
        r"SOFT TISSUES"
    ],
    "Composition": [r"BREAST COMPOSITION"],
    "Impression": [r"IMPRESSION", r"IMP"],
    "Assessment": [r"ASSESSMENT", r"FINAL ASSESSMENT", r"ASSESSSMENT"],
    "Recommendation": [r"RECOMMENDATION", r"PLAN"],
    "Procedure": [r"PROCEDURE"]
}

KNOWN_MODALITIES = {
    "us": ["us", "ultrasound", "sonography"],
    "ct": ["ct", "computed tomography"],
    "mri": ["mri", "mr", "magnetic resonance"],
    "xray": ["xray", "x-ray", "xr", "radiograph"],
    "mammogram": ["mammogram", "mammography", "mg"]
}

KNOWN_ORGANS = [
    "right thyroid lobe",
    "left thyroid lobe",
    "isthmus",
    "thyroid",
    "cervical lymph nodes",
    "lymph nodes",
    "liver",
    "gallbladder",
    "pancreas",
    "spleen",
    "kidney",
    "kidneys",
    "adrenal",
    "adrenals",
    "bladder",
    "bowel loops",
    "vasculature",
    "lung",
    "lungs",
    "pleura",
    "heart",
    "mediastinum",
    "hilar structures"
]

KNOWN_FINDINGS = [
    "nodule",
    "cyst",
    "microcalcifications",
    "microcalcification",
    "enlarged",
    "lesion",
    "mass",
    "calcification",
    "hypoechoic",
    "heterogeneous",
    "normal",
    "unremarkable"
]

STOPWORDS = {
    "show", "find", "for", "with", "and", "the", "a", "an", "of", "in", "on",
    "patients", "patient", "reports", "report", "results", "result", "please",
    "all", "only", "me", "give", "display", "search", "than", "more", "over",
    "study", "studies", "case", "cases", "from", "to", "by", "is", "are"
}

STUDY_TEMPLATES = {
    "abdomen": {
        "organs": ["liver", "gallbladder", "spleen", "pancreas", "kidneys", "adrenals", "bladder", "bowel loops", "vasculature"],
        "default_text": "Unremarkable."
    },
    "chest": {
        "organs": ["lungs", "pleura", "heart", "mediastinum", "hilar structures"],
        "default_text": "Unremarkable."
    },
    "thyroid": {
        "organs": ["right thyroid lobe", "left thyroid lobe", "isthmus", "cervical lymph nodes"],
        "default_text": "No focal abnormality identified."
    }
}


# =========================================================
# HELPERS
# =========================================================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\s\r\n\t]+", " ", text)
    return text


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(str(value))
    return result


def safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def parse_size_to_cm(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)\b", str(text), re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value / 10.0 if unit == "mm" else value


def truncate_text(text: str, max_len: int = 800) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def modality_to_study_bucket(modality_text: str, exam_text: str) -> Optional[str]:
    combined = normalize_text(f"{modality_text} {exam_text}")
    if any(k in combined for k in ["abdomen", "hepato", "liver", "gallbladder", "pancrea", "renal", "kidney"]):
        return "abdomen"
    if any(k in combined for k in ["chest", "lung", "pleura", "thorax", "mediast"]):
        return "chest"
    if "thyroid" in combined or "neck" in combined:
        return "thyroid"
    return None


def format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# =========================================================
# HARDWARE
# =========================================================
class HardwareManager:
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.info("Using MPS")
            return torch.device("mps")
        logger.info("Using CPU")
        return torch.device("cpu")

    @staticmethod
    def get_mixed_precision_context():
        device = HardwareManager.get_device()
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()


# =========================================================
# ONTOLOGY
# =========================================================
class OntologyMapper:
    def __init__(self):
        self.anatomy_map = {
            "thyroid": ("RadLex", "RID58"),
            "right thyroid lobe": ("RadLex", "RID59"),
            "left thyroid lobe": ("RadLex", "RID60"),
            "isthmus": ("RadLex", "RID61"),
            "liver": ("RadLex", "RID10224"),
            "gallbladder": ("RadLex", "RID187"),
            "pancreas": ("RadLex", "RID170"),
            "spleen": ("RadLex", "RID86"),
            "lymph nodes": ("RadLex", "RID284"),
            "cervical lymph nodes": ("RadLex", "RID296"),
            "lung": ("RadLex", "RID1301"),
            "lungs": ("RadLex", "RID1301"),
            "pleura": ("RadLex", "RID15263"),
            "heart": ("RadLex", "RID1385"),
            "kidney": ("RadLex", "RID1305"),
            "kidneys": ("RadLex", "RID1305"),
            "bladder": ("RadLex", "RID1310"),
            "mediastinum": ("RadLex", "RID1362"),
        }

        self.findings_map = {
            "nodule": ("RadLex", "RID3874"),
            "cyst": ("RadLex", "RID3887"),
            "microcalcifications": ("RadLex", "RID3464"),
            "microcalcification": ("RadLex", "RID3464"),
            "enlarged": ("SNOMED", "26036001"),
            "lesion": ("RadLex", "RID5017"),
            "mass": ("RadLex", "RID2846"),
            "calcification": ("RadLex", "RID3464"),
            "hypoechoic": ("SNOMED", "260385009"),
            "heterogeneous": ("SNOMED", "260381009"),
            "normal": ("SNOMED", "17621005"),
            "unremarkable": ("SNOMED", "17621005"),
        }

    def map_anatomy(self, term: str) -> Tuple[str, str]:
        t = normalize_text(term)
        if t in self.anatomy_map:
            return self.anatomy_map[t]
        for key, val in self.anatomy_map.items():
            if key in t:
                return val
        return ("Unknown", "UNK")

    def map_finding(self, term: str) -> Tuple[str, str]:
        t = normalize_text(term)
        if t in self.findings_map:
            return self.findings_map[t]
        for key, val in self.findings_map.items():
            if key in t:
                return val
        return ("Unknown", "UNK")


# =========================================================
# DATA LOADER
# =========================================================
class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _pick(self, record: Dict[str, Any], *keys, default=None):
        for key in keys:
            if key in record and record.get(key) not in (None, ""):
                return record.get(key)
        return default

    def load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file/directory not found: {self.data_path}")

        rows: List[Dict[str, Any]] = []

        if os.path.isdir(self.data_path):
            for file_name in os.listdir(self.data_path):
                if file_name.lower().endswith(".json"):
                    file_path = os.path.join(self.data_path, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            rows.extend(data)
                        else:
                            rows.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
            else:
                rows.append(data)

        logger.info(f"Loaded {len(rows)} raw records")
        return self._preprocess(rows)

    def _preprocess(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []

        for index, raw in enumerate(rows):
            try:
                record = dict(raw)
                gender_raw = str(self._pick(record, "GENDER", default="") or "").strip().upper()
                if gender_raw == "M":
                    gender = "Male"
                elif gender_raw == "F":
                    gender = "Female"
                elif gender_raw in ("MALE", "FEMALE"):
                    gender = gender_raw.title()
                else:
                    gender = "Unknown"

                item = {
                    "HN": self._pick(record, "HN", "hn", default=""),
                    "PATIENT_NAME": self._pick(record, "PATIENT_NAME", "PATIENTNAME", default=""),
                    "GENDER": gender,
                    "DOB": self._pick(record, "DOB", default=""),
                    "PATIENT_AGE": safe_int(self._pick(record, "PATIENT_AGE", "PATIENTAGE", default=0), 0),
                    "EMP_UID": self._pick(record, "EMP_UID", "EMPUID", default=""),
                    "DOCTOR_NAME": self._pick(record, "DOCTOR_NAME", "DOCTORNAME", default=""),
                    "MODALITY_TYPE": self._pick(record, "MODALITY_TYPE", "MODALITYTYPE", default=""),
                    "MODALITY": self._pick(record, "MODALITY", default=""),
                    "ACCESSION_NO": self._pick(record, "ACCESSION_NO", "ACCESSIONNO", default=""),
                    "ORDER_DT": self._pick(record, "ORDER_DT", "ORDERDT", default=""),
                    "EXAM_DT": self._pick(record, "EXAM_DT", "EXAMDT", default=""),
                    "SEVERITY_NAME": self._pick(record, "SEVERITY_NAME", "SEVERITYNAME", default=""),
                    "EXAM_UID": self._pick(record, "EXAM_UID", "EXAMUID", default=""),
                    "EXAM_NAME": self._pick(record, "EXAM_NAME", "EXAMNAME", default=""),
                    "EXAM_TYPE_TEXT": self._pick(record, "EXAM_TYPE_TEXT", "EXAMTYPETEXT", default=""),
                    "CLINICAL_INSTRUCTION": self._pick(record, "CLINICAL_INSTRUCTION", "CLINICALINSTRUCTION", default="") or "",
                    "RESULT_TEXT_PLAIN": self._pick(record, "RESULT_TEXT_PLAIN", "RESULTTEXTPLAIN", default="") or "",
                }
                processed.append(item)
            except Exception as e:
                logger.warning(f"Preprocess failed at row {index}: {e}")

        logger.info(f"Prepared {len(processed)} normalized records")
        return processed


# =========================================================
# REPORT PARSER
# =========================================================
class ReportParser:
    def __init__(self, aliases: Optional[Dict[str, list]] = None):
        self.aliases = aliases or HEADER_ALIASES

    def parse_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        if not text:
            return sections

        header_patterns = []
        for header, patterns in self.aliases.items():
            for idx, pattern in enumerate(patterns):
                header_patterns.append(f"(?P<{header}_{idx}>{pattern}\\s*:)")

        combined_pattern = "|".join(header_patterns)
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE | re.MULTILINE))

        if not matches:
            sections["unstructured"] = text.strip()
            return sections

        if matches[0].start() > 0:
            sections["Header"] = text[:matches[0].start()].strip()

        for i, match in enumerate(matches):
            header_name = match.lastgroup.split("_")[0]
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_idx:end_idx].strip()
            if content:
                sections[header_name] = (sections.get(header_name, "") + "\n" + content).strip()

        return sections


# =========================================================
# ENTITY EXTRACTION
# =========================================================
class EntityExtractor:
    def __init__(self):
        self.ontology_mapper = OntologyMapper()
        self.meas_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:[xX\*]\s*(\d+(?:\.\d+)?))?\s*(?:[xX\*]\s*(\d+(?:\.\d+)?))?\s*(cm|mm)",
            re.IGNORECASE,
        )
        self.anatomies = sorted(KNOWN_ORGANS, key=len, reverse=True)
        self.findings = sorted(KNOWN_FINDINGS, key=len, reverse=True)

    def _to_cm(self, dims: List[float], unit: str) -> List[float]:
        if normalize_text(unit) == "mm":
            return [round(d / 10.0, 4) for d in dims]
        return [round(d, 4) for d in dims]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if not text or not text.strip():
            return entities

        text_norm = normalize_text(text)
        measurements = []

        for match in self.meas_pattern.finditer(text):
            dims = [float(d) for d in match.groups()[:-1] if d is not None]
            unit = match.groups()[-1].lower()
            dims_cm = self._to_cm(dims, unit)
            measurements.append({
                "type": "Measurement",
                "value": " x ".join(map(str, dims)) + f" {unit}",
                "dimensions": dims,
                "dimensions_cm": dims_cm,
                "max_dimension_cm": max(dims_cm) if dims_cm else None,
                "unit": unit,
                "raw_text": match.group().strip()
            })

        found_anatomies = []
        for anat in self.anatomies:
            if normalize_text(anat) in text_norm:
                onto_db, onto_id = self.ontology_mapper.map_anatomy(anat)
                found_anatomies.append({
                    "type": "Organ",
                    "name": anat,
                    "name_normalized": normalize_text(anat),
                    "ontology": onto_db,
                    "concept_id": onto_id
                })

        found_findings = []
        for term in self.findings:
            if normalize_text(term) in text_norm:
                onto_db, onto_id = self.ontology_mapper.map_finding(term)
                found_findings.append({
                    "type": "FindingTerm",
                    "name": term,
                    "name_normalized": normalize_text(term),
                    "ontology": onto_db,
                    "concept_id": onto_id
                })

        found_anatomies = [dict(t) for t in {tuple(sorted(x.items())) for x in found_anatomies}]
        found_findings = [dict(t) for t in {tuple(sorted(x.items())) for x in found_findings}]

        if found_anatomies or found_findings or measurements:
            entities.append({
                "context_block": text.strip(),
                "context_block_normalized": text_norm,
                "organs": found_anatomies,
                "finding_terms": found_findings,
                "measurements": measurements
            })

        return entities

    def process_sections(self, sections: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        section_entities = {}
        for section_name, content in sections.items():
            if section_name.lower() in ["findings", "impression", "assessment", "unstructured"]:
                lines = [line.strip(" -•\t") for line in content.split("\n") if line.strip()]
                entities = []
                for line in lines:
                    extracted = self.extract_entities(line)
                    if extracted:
                        entities.extend(extracted)

                if not entities and content.strip():
                    entities = [{
                        "context_block": content.strip(),
                        "context_block_normalized": normalize_text(content),
                        "organs": [],
                        "finding_terms": [],
                        "measurements": []
                    }]

                if entities:
                    section_entities[section_name] = entities
        return section_entities


# =========================================================
# QUERY PARSING
# =========================================================
class DeterministicQueryParser:
    def __init__(self):
        self.measurement_min_pattern = re.compile(
            r"(?:more than|greater than|over|at least|min(?:imum)?|>=?)\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
            re.IGNORECASE
        )
        self.measurement_max_pattern = re.compile(
            r"(?:less than|under|below|at most|max(?:imum)?|<=?)\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
            re.IGNORECASE
        )

    def _extract_gender(self, query: str) -> Optional[str]:
        q = query.lower()
        if re.search(r"\bfemale\b|\bwoman\b|\bwomen\b", q):
            return "Female"
        if re.search(r"\bmale\b|\bman\b|\bmen\b", q):
            return "Male"
        return None

    def _extract_age_range(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        q = query.lower()
        min_age, max_age = None, None

        m = re.search(r"(?:between|from)\s+(\d{1,3})\s+(?:and|to)\s+(\d{1,3})", q)
        if m:
            return int(m.group(1)), int(m.group(2))

        m = re.search(r"(?:over|older than|above|greater than)\s+(\d{1,3})", q)
        if m:
            min_age = int(m.group(1)) + 1

        m = re.search(r"(?:at least|min(?:imum)?)\s+(\d{1,3})", q)
        if m and min_age is None:
            min_age = int(m.group(1))

        m = re.search(r"(?:under|younger than|below|less than)\s+(\d{1,3})", q)
        if m:
            max_age = int(m.group(1)) - 1

        m = re.search(r"(?:at most|max(?:imum)?)\s+(\d{1,3})", q)
        if m and max_age is None:
            max_age = int(m.group(1))

        return min_age, max_age

    def _extract_modality(self, query: str) -> Optional[str]:
        q = normalize_text(query)
        for canonical, aliases in KNOWN_MODALITIES.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", q):
                    return canonical
        return None

    def _extract_term(self, query: str, candidates: List[str]) -> Optional[str]:
        q = normalize_text(query)
        for candidate in sorted(candidates, key=len, reverse=True):
            if re.search(rf"\b{re.escape(normalize_text(candidate))}\b", q):
                return candidate
        return None

    def _extract_measurement(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        q = query.lower()
        min_cm = None
        max_cm = None

        m = self.measurement_min_pattern.search(q)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            min_cm = val / 10.0 if unit == "mm" else val

        m = self.measurement_max_pattern.search(q)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            max_cm = val / 10.0 if unit == "mm" else val

        return min_cm, max_cm

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", query.lower())
        return unique_preserve_order([t for t in tokens if t not in STOPWORDS and len(t) > 2])[:12]

    def parse_query(self, query: str) -> dict:
        min_age, max_age = self._extract_age_range(query)
        min_dim, max_dim = self._extract_measurement(query)
        result = {
            "gender": self._extract_gender(query),
            "min_age": min_age,
            "max_age": max_age,
            "modality": self._extract_modality(query),
            "target_organ": self._extract_term(query, KNOWN_ORGANS),
            "finding": self._extract_term(query, KNOWN_FINDINGS),
            "min_dimension_cm": min_dim,
            "max_dimension_cm": max_dim,
            "keywords": self._extract_keywords(query)
        }
        return {k: v for k, v in result.items() if v not in (None, "", [], {})}


class QueryParserLLM:
    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.url = url
        self.model = model
        self.rule_parser = DeterministicQueryParser()

    def _normalize_llm_result(self, data: dict) -> dict:
        result = {}
        for key in [
            "gender", "min_age", "max_age", "modality", "target_organ",
            "finding", "min_dimension_cm", "max_dimension_cm", "keywords"
        ]:
            if key in data and data[key] not in (None, "", [], {}):
                result[key] = data[key]

        if "min_age" in result:
            result["min_age"] = safe_int(result["min_age"], None)
        if "max_age" in result:
            result["max_age"] = safe_int(result["max_age"], None)
        if "min_dimension_cm" in result:
            result["min_dimension_cm"] = safe_float(result["min_dimension_cm"], None)
        if "max_dimension_cm" in result:
            result["max_dimension_cm"] = safe_float(result["max_dimension_cm"], None)

        if "gender" in result:
            g = normalize_text(result["gender"])
            if g == "female":
                result["gender"] = "Female"
            elif g == "male":
                result["gender"] = "Male"

        if "keywords" in result and isinstance(result["keywords"], str):
            result["keywords"] = [result["keywords"]]

        return result

    def parse_query(self, query: str) -> dict:
        rule_based = self.rule_parser.parse_query(query)
        if not query.strip():
            return rule_based

        prompt = f"""
You are a medical informatics expert.
Parse the following clinical query into strict JSON only.

Allowed keys:
- gender
- min_age
- max_age
- modality
- target_organ
- finding
- min_dimension_cm
- max_dimension_cm
- keywords

Rules:
- Return JSON only.
- Omit absent keys.
- Convert mm to cm.
- keywords must be an array of short terms.

Query: "{query}"
""".strip()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        llm_result = {}
        try:
            response = requests.post(f"{self.url}/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            text = response.json().get("response", "{}")
            llm_result = self._normalize_llm_result(json.loads(text))
        except Exception as e:
            logger.warning(f"LLM parse failed, using rule-based parse only: {e}")

        merged = dict(llm_result)
        for k, v in rule_based.items():
            if merged.get(k) in (None, "", [], {}):
                merged[k] = v

        if not merged.get("keywords"):
            merged["keywords"] = rule_based.get("keywords", [])

        return merged


# =========================================================
# EMBEDDINGS
# =========================================================
class EmbeddingGenerator:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.device = HardwareManager.get_device()
        self.model = None
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device=str(self.device))
            logger.info(f"Embedding model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Embedding model unavailable: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts or self.model is None:
            return []
        with HardwareManager.get_mixed_precision_context():
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        if self.model is None:
            return 768
        return int(self.model.get_sentence_embedding_dimension())


# =========================================================
# GRAPH BUILDER
# =========================================================
class GraphBuilder:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify(self) -> Tuple[bool, str]:
        try:
            with self.driver.session() as session:
                rec = session.run("RETURN 1 AS ok").single()
                if rec and rec["ok"] == 1:
                    return True, "Neo4j connected"
            return False, "Neo4j verification failed"
        except Exception as e:
            return False, f"Neo4j error: {e}"

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return list(session.run(query, **params))

    def setup_constraints(self):
        statements = [
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT study_id IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT exam_id IF NOT EXISTS FOR (e:Exam) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT doctor_id IF NOT EXISTS FOR (d:Doctor) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT organ_key IF NOT EXISTS FOR (o:Organ) REQUIRE o.key IS UNIQUE",
            "CREATE CONSTRAINT finding_term_key IF NOT EXISTS FOR (ft:FindingTerm) REQUIRE ft.key IS UNIQUE",
            "CREATE CONSTRAINT measurement_id IF NOT EXISTS FOR (m:Measurement) REQUIRE m.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(f"Constraint skipped: {e}")

    def setup_indexes(self):
        statements = [
            "CREATE INDEX patient_gender IF NOT EXISTS FOR (p:Patient) ON (p.gender)",
            "CREATE INDEX patient_age IF NOT EXISTS FOR (p:Patient) ON (p.age)",
            "CREATE INDEX study_severity_normalized IF NOT EXISTS FOR (s:Study) ON (s.severity_normalized)",
            "CREATE INDEX study_clinical_instruction_normalized IF NOT EXISTS FOR (s:Study) ON (s.clinical_instruction_normalized)",
            "CREATE INDEX exam_name_normalized IF NOT EXISTS FOR (e:Exam) ON (e.name_normalized)",
            "CREATE INDEX modality_name_normalized IF NOT EXISTS FOR (m:Modality) ON (m.name_normalized)",
            "CREATE INDEX modality_type_normalized IF NOT EXISTS FOR (m:Modality) ON (m.type_normalized)",
            "CREATE INDEX finding_text_normalized IF NOT EXISTS FOR (f:Finding) ON (f.text_normalized)",
            "CREATE INDEX organ_name_normalized IF NOT EXISTS FOR (o:Organ) ON (o.name_normalized)",
            "CREATE INDEX finding_term_name_normalized IF NOT EXISTS FOR (ft:FindingTerm) ON (ft.name_normalized)",
            "CREATE INDEX measurement_max_dimension_cm IF NOT EXISTS FOR (m:Measurement) ON (m.max_dimension_cm)",
            "CREATE FULLTEXT INDEX finding_text_ft IF NOT EXISTS FOR (f:Finding) ON EACH [f.text, f.text_normalized]",
            "CREATE FULLTEXT INDEX study_text_ft IF NOT EXISTS FOR (s:Study) ON EACH [s.clinical_instruction, s.clinical_instruction_normalized, s.severity, s.severity_normalized]",
            "CREATE FULLTEXT INDEX exam_text_ft IF NOT EXISTS FOR (e:Exam) ON EACH [e.name, e.name_normalized, e.type, e.type_normalized]"
        ]
        with self.driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(f"Index skipped: {e}")

    def ingest_report(self, record: Dict[str, Any], extracted_entities: Dict[str, List[Dict[str, Any]]] = None):
        query = """
        MERGE (p:Patient {id: $hn})
        ON CREATE SET
            p.name = $patient_name,
            p.gender = $gender,
            p.gender_normalized = $gender_normalized,
            p.dob = $dob,
            p.age = $age
        ON MATCH SET
            p.name = coalesce($patient_name, p.name),
            p.gender = coalesce($gender, p.gender),
            p.gender_normalized = coalesce($gender_normalized, p.gender_normalized),
            p.dob = coalesce($dob, p.dob),
            p.age = coalesce($age, p.age)

        MERGE (d:Doctor {id: $emp_uid})
        ON CREATE SET d.name = $doctor_name
        ON MATCH SET d.name = coalesce($doctor_name, d.name)

        MERGE (m:Modality {type: $modality_type})
        ON CREATE SET
            m.name = $modality_name,
            m.name_normalized = $modality_name_normalized,
            m.type_normalized = $modality_type_normalized
        ON MATCH SET
            m.name = coalesce($modality_name, m.name),
            m.name_normalized = coalesce($modality_name_normalized, m.name_normalized),
            m.type_normalized = coalesce($modality_type_normalized, m.type_normalized)

        MERGE (s:Study {id: $accession_no})
        ON CREATE SET
            s.order_dt = $order_dt,
            s.exam_dt = $exam_dt,
            s.severity = $severity,
            s.severity_normalized = $severity_normalized,
            s.clinical_instruction = $clinical_instruction,
            s.clinical_instruction_normalized = $clinical_instruction_normalized,
            s.result_text_plain = $result_text_plain,
            s.result_text_plain_normalized = $result_text_plain_normalized
        ON MATCH SET
            s.order_dt = coalesce($order_dt, s.order_dt),
            s.exam_dt = coalesce($exam_dt, s.exam_dt),
            s.severity = coalesce($severity, s.severity),
            s.severity_normalized = coalesce($severity_normalized, s.severity_normalized),
            s.clinical_instruction = coalesce($clinical_instruction, s.clinical_instruction),
            s.clinical_instruction_normalized = coalesce($clinical_instruction_normalized, s.clinical_instruction_normalized),
            s.result_text_plain = coalesce($result_text_plain, s.result_text_plain),
            s.result_text_plain_normalized = coalesce($result_text_plain_normalized, s.result_text_plain_normalized)

        MERGE (e:Exam {id: $exam_uid})
        ON CREATE SET
            e.name = $exam_name,
            e.name_normalized = $exam_name_normalized,
            e.type = $exam_type_text,
            e.type_normalized = $exam_type_text_normalized
        ON MATCH SET
            e.name = coalesce($exam_name, e.name),
            e.name_normalized = coalesce($exam_name_normalized, e.name_normalized),
            e.type = coalesce($exam_type_text, e.type),
            e.type_normalized = coalesce($exam_type_text_normalized, e.type_normalized)

        MERGE (c:ClinicalInstruction {text: $clinical_instruction})
        ON CREATE SET c.text_normalized = $clinical_instruction_normalized
        ON MATCH SET c.text_normalized = coalesce($clinical_instruction_normalized, c.text_normalized)

        MERGE (p)-[:HAS_STUDY]->(s)
        MERGE (s)-[:PERFORMED_BY]->(d)
        MERGE (s)-[:HAS_MODALITY]->(m)
        MERGE (s)-[:HAS_EXAM]->(e)
        MERGE (s)-[:HAS_CLINICAL_REASON]->(c)
        """

        params = {
            "hn": record.get("HN") or "",
            "patient_name": record.get("PATIENT_NAME") or "",
            "gender": record.get("GENDER") or "",
            "gender_normalized": normalize_text(record.get("GENDER") or ""),
            "dob": record.get("DOB") or "",
            "age": safe_int(record.get("PATIENT_AGE"), 0),
            "emp_uid": record.get("EMP_UID") or "",
            "doctor_name": record.get("DOCTOR_NAME") or "",
            "modality_type": record.get("MODALITY_TYPE") or "",
            "modality_name": record.get("MODALITY") or "",
            "modality_name_normalized": normalize_text(record.get("MODALITY") or ""),
            "modality_type_normalized": normalize_text(record.get("MODALITY_TYPE") or ""),
            "accession_no": record.get("ACCESSION_NO") or "",
            "order_dt": record.get("ORDER_DT") or "",
            "exam_dt": record.get("EXAM_DT") or "",
            "severity": record.get("SEVERITY_NAME") or "",
            "severity_normalized": normalize_text(record.get("SEVERITY_NAME") or ""),
            "exam_uid": record.get("EXAM_UID") or (record.get("ACCESSION_NO") or ""),
            "exam_name": record.get("EXAM_NAME") or "",
            "exam_name_normalized": normalize_text(record.get("EXAM_NAME") or ""),
            "exam_type_text": record.get("EXAM_TYPE_TEXT") or "",
            "exam_type_text_normalized": normalize_text(record.get("EXAM_TYPE_TEXT") or ""),
            "clinical_instruction": record.get("CLINICAL_INSTRUCTION") or "Unspecified",
            "clinical_instruction_normalized": normalize_text(record.get("CLINICAL_INSTRUCTION") or "Unspecified"),
            "result_text_plain": record.get("RESULT_TEXT_PLAIN") or "",
            "result_text_plain_normalized": normalize_text(record.get("RESULT_TEXT_PLAIN") or ""),
        }

        with self.driver.session() as session:
            session.run(query, **params)

            if not extracted_entities:
                extracted_entities = {
                    "unstructured": [{
                        "context_block": record.get("RESULT_TEXT_PLAIN", "") or "",
                        "context_block_normalized": normalize_text(record.get("RESULT_TEXT_PLAIN", "") or ""),
                        "organs": [],
                        "finding_terms": [],
                        "measurements": []
                    }]
                }

            for section, entities in extracted_entities.items():
                for idx, entity in enumerate(entities):
                    self._ingest_entity(session, record.get("ACCESSION_NO") or "", section, idx, entity)

    def _ingest_entity(self, session, study_id: str, section: str, idx: int, entity: Dict[str, Any]):
        finding_id = f"{study_id}_{section}_{idx}"

        finding_query = """
        MATCH (s:Study {id: $study_id})
        MERGE (f:Finding {id: $finding_id})
        ON CREATE SET
            f.text = $context_block,
            f.text_normalized = $context_block_normalized,
            f.section = $section
        ON MATCH SET
            f.text = coalesce($context_block, f.text),
            f.text_normalized = coalesce($context_block_normalized, f.text_normalized),
            f.section = coalesce($section, f.section)
        MERGE (s)-[:HAS_FINDING]->(f)
        """
        session.run(
            finding_query,
            study_id=study_id,
            finding_id=finding_id,
            context_block=entity.get("context_block", ""),
            context_block_normalized=entity.get("context_block_normalized", ""),
            section=section
        )

        for organ in entity.get("organs", []):
            organ_query = """
            MATCH (f:Finding {id: $finding_id})
            MERGE (o:Organ {key: $key})
            ON CREATE SET
                o.name = $name,
                o.name_normalized = $name_normalized,
                o.concept_id = $concept_id,
                o.ontology = $ontology
            ON MATCH SET
                o.name = coalesce($name, o.name),
                o.name_normalized = coalesce($name_normalized, o.name_normalized),
                o.concept_id = coalesce($concept_id, o.concept_id),
                o.ontology = coalesce($ontology, o.ontology)
            MERGE (f)-[:LOCATED_IN]->(o)
            """
            session.run(
                organ_query,
                finding_id=finding_id,
                key=organ["name_normalized"],
                name=organ["name"],
                name_normalized=organ["name_normalized"],
                concept_id=organ["concept_id"],
                ontology=organ["ontology"]
            )

        for term in entity.get("finding_terms", []):
            term_query = """
            MATCH (f:Finding {id: $finding_id})
            MERGE (ft:FindingTerm {key: $key})
            ON CREATE SET
                ft.name = $name,
                ft.name_normalized = $name_normalized,
                ft.concept_id = $concept_id,
                ft.ontology = $ontology
            ON MATCH SET
                ft.name = coalesce($name, ft.name),
                ft.name_normalized = coalesce($name_normalized, ft.name_normalized),
                ft.concept_id = coalesce($concept_id, ft.concept_id),
                ft.ontology = coalesce($ontology, ft.ontology)
            MERGE (f)-[:HAS_FINDING_TERM]->(ft)
            """
            session.run(
                term_query,
                finding_id=finding_id,
                key=term["name_normalized"],
                name=term["name"],
                name_normalized=term["name_normalized"],
                concept_id=term["concept_id"],
                ontology=term["ontology"]
            )

        for meas_idx, meas in enumerate(entity.get("measurements", [])):
            meas_query = """
            MATCH (f:Finding {id: $finding_id})
            MERGE (m:Measurement {id: $measurement_id})
            ON CREATE SET
                m.value = $value,
                m.raw_text = $raw_text,
                m.unit = $unit,
                m.dimensions = $dimensions,
                m.dimensions_cm = $dimensions_cm,
                m.max_dimension_cm = $max_dimension_cm
            ON MATCH SET
                m.value = coalesce($value, m.value),
                m.raw_text = coalesce($raw_text, m.raw_text),
                m.unit = coalesce($unit, m.unit),
                m.dimensions = coalesce($dimensions, m.dimensions),
                m.dimensions_cm = coalesce($dimensions_cm, m.dimensions_cm),
                m.max_dimension_cm = coalesce($max_dimension_cm, m.max_dimension_cm)
            MERGE (f)-[:HAS_MEASUREMENT]->(m)
            """
            session.run(
                meas_query,
                finding_id=finding_id,
                measurement_id=f"{finding_id}_m_{meas_idx}",
                value=meas.get("value"),
                raw_text=meas.get("raw_text"),
                unit=meas.get("unit"),
                dimensions=meas.get("dimensions", []),
                dimensions_cm=meas.get("dimensions_cm", []),
                max_dimension_cm=meas.get("max_dimension_cm")
            )

    def get_filter_options(self, search_prefix: str = "", limit: int = 100) -> Dict[str, List[str]]:
        prefix = normalize_text(search_prefix)
        query = """
        CALL {
            MATCH (e:Exam)
            WHERE $prefix = "" OR toLower(coalesce(e.name_normalized, "")) CONTAINS $prefix
            RETURN collect(DISTINCT e.name)[0..$limit] AS vals
        }
        RETURN vals
        """
        def fetch_list(cypher: str) -> List[str]:
            try:
                with self.driver.session() as session:
                    rec = session.run(cypher, prefix=prefix, limit=limit).single()
                    return [x for x in (rec["vals"] if rec else []) if x]
            except Exception:
                return []

        exams = fetch_list(query)

        queries = {
            "organs": """
                MATCH (o:Organ)
                WHERE $prefix = "" OR toLower(coalesce(o.name_normalized, "")) CONTAINS $prefix
                RETURN collect(DISTINCT o.name)[0..$limit] AS vals
            """,
            "findings": """
                MATCH (ft:FindingTerm)
                WHERE $prefix = "" OR toLower(coalesce(ft.name_normalized, "")) CONTAINS $prefix
                RETURN collect(DISTINCT ft.name)[0..$limit] AS vals
            """,
            "modalities": """
                MATCH (m:Modality)
                WHERE $prefix = "" OR toLower(coalesce(m.name_normalized, "")) CONTAINS $prefix OR toLower(coalesce(m.type_normalized, "")) CONTAINS $prefix
                RETURN collect(DISTINCT coalesce(m.name, m.type))[0..$limit] AS vals
            """,
            "severity": """
                MATCH (s:Study)
                WHERE $prefix = "" OR toLower(coalesce(s.severity_normalized, "")) CONTAINS $prefix
                RETURN collect(DISTINCT s.severity)[0..$limit] AS vals
            """,
            "history": """
                MATCH (s:Study)
                WHERE $prefix = "" OR toLower(coalesce(s.clinical_instruction_normalized, "")) CONTAINS $prefix
                RETURN collect(DISTINCT s.clinical_instruction)[0..$limit] AS vals
            """
        }

        data = {"exams": exams}
        for key, q in queries.items():
            data[key] = fetch_list(q)

        return {k: unique_preserve_order(v) for k, v in data.items()}


# =========================================================
# VECTOR INDEX
# =========================================================
class VectorIndexBuilder:
    def __init__(self, graph_builder: GraphBuilder):
        self.gb = graph_builder

    def create_finding_vector_index(self, vector_dimension: int = 768):
        query = f"""
        CREATE VECTOR INDEX finding_text_embeddings IF NOT EXISTS
        FOR (f:Finding) ON (f.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {vector_dimension},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        try:
            with self.gb.driver.session() as session:
                session.run(query)
            logger.info("Vector index ready")
        except Exception as e:
            logger.warning(f"Vector index creation skipped: {e}")

    def update_finding_embeddings(self, finding_id: str, embedding: List[float]):
        query = """
        MATCH (f:Finding {id: $finding_id})
        SET f.embedding = $embedding
        """
        try:
            with self.gb.driver.session() as session:
                session.run(query, finding_id=finding_id, embedding=embedding)
        except Exception as e:
            logger.warning(f"Embedding update failed for {finding_id}: {e}")


# =========================================================
# SEARCH
# =========================================================
class HybridSearch:
    def __init__(self, graph_builder: GraphBuilder, embedding_generator: EmbeddingGenerator):
        self.gb = graph_builder
        self.eg = embedding_generator

    def _merge_filters(self, parsed_query: dict, advanced_filters: Optional[dict]) -> dict:
        merged = dict(parsed_query or {})
        af = advanced_filters or {}

        for k in ["exam_name", "accession_no", "severity", "clinical_history", "modality", "target_organ", "finding"]:
            if af.get(k) not in (None, "", [], {}):
                merged[k] = af[k]

        if af.get("organ_name"):
            merged["target_organ"] = af["organ_name"]

        if af.get("min_age") is not None:
            merged["min_age"] = af["min_age"] if merged.get("min_age") is None else max(int(merged["min_age"]), int(af["min_age"]))

        if af.get("max_age") is not None:
            merged["max_age"] = af["max_age"] if merged.get("max_age") is None else min(int(merged["max_age"]), int(af["max_age"]))

        if af.get("min_dimension_cm") is not None:
            merged["min_dimension_cm"] = float(af["min_dimension_cm"])

        if af.get("max_dimension_cm") is not None:
            merged["max_dimension_cm"] = float(af["max_dimension_cm"])

        return merged

    def _derive_keywords(self, merged: dict, raw_query: str) -> List[str]:
        terms = []

        if isinstance(merged.get("keywords"), list):
            terms.extend(merged["keywords"])

        for key in ["target_organ", "finding", "modality", "exam_name", "clinical_history", "severity"]:
            val = merged.get(key)
            if val:
                if isinstance(val, list):
                    terms.extend(map(str, val))
                else:
                    terms.append(str(val))

        if not terms:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", raw_query.lower())
            terms.extend([t for t in tokens if t not in STOPWORDS and len(t) > 2])

        return unique_preserve_order([normalize_text(t) for t in terms if normalize_text(t)])[:15]

    def _base_params(self, merged: dict, raw_query: str, top_k: int) -> dict:
        keywords = self._derive_keywords(merged, raw_query)
        return {
            "top_k": int(top_k),
            "gender": normalize_text(merged.get("gender", "")),
            "min_age": merged.get("min_age"),
            "max_age": merged.get("max_age"),
            "modality": normalize_text(merged.get("modality", "")),
            "organ": normalize_text(merged.get("target_organ", "")),
            "finding": normalize_text(merged.get("finding", "")),
            "exam_name": normalize_text(merged.get("exam_name", "")),
            "accession_no": merged.get("accession_no", ""),
            "severity": normalize_text(merged.get("severity", "")),
            "clinical_history": normalize_text(merged.get("clinical_history", "")),
            "min_dimension_cm": safe_float(merged.get("min_dimension_cm"), None),
            "max_dimension_cm": safe_float(merged.get("max_dimension_cm"), None),
            "keywords": keywords,
            "raw_query": normalize_text(raw_query),
        }

    def _graph_search(self, params: dict) -> List[Dict[str, Any]]:
        graph_query = """
        MATCH (p:Patient)-[:HAS_STUDY]->(s:Study)
        OPTIONAL MATCH (s)-[:HAS_EXAM]->(e:Exam)
        OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality)
        OPTIONAL MATCH (s)-[:HAS_FINDING]->(f:Finding)
        OPTIONAL MATCH (f)-[:LOCATED_IN]->(o:Organ)
        OPTIONAL MATCH (f)-[:HAS_FINDING_TERM]->(ft:FindingTerm)
        OPTIONAL MATCH (f)-[:HAS_MEASUREMENT]->(meas:Measurement)

        WITH p, s, e, m, f,
             collect(DISTINCT o.name) AS organs,
             collect(DISTINCT o.name_normalized) AS organs_norm,
             collect(DISTINCT ft.name) AS finding_terms,
             collect(DISTINCT ft.name_normalized) AS finding_terms_norm,
             collect(DISTINCT meas.raw_text) AS measurement_texts,
             max(meas.max_dimension_cm) AS max_dimension_cm,
             coalesce(f.text_normalized, s.result_text_plain_normalized, "") AS searchable_finding_text,
             coalesce(s.clinical_instruction_normalized, "") AS searchable_history,
             coalesce(e.name_normalized, "") AS searchable_exam,
             coalesce(e.type_normalized, "") AS searchable_exam_type,
             coalesce(m.name_normalized, "") AS searchable_modality_name,
             coalesce(m.type_normalized, "") AS searchable_modality_type

        WHERE
            ($gender = "" OR toLower(coalesce(p.gender, "")) = $gender)
            AND ($min_age IS NULL OR toInteger(coalesce(p.age, 0)) >= $min_age)
            AND ($max_age IS NULL OR toInteger(coalesce(p.age, 0)) <= $max_age)
            AND ($accession_no = "" OR s.id = $accession_no)
            AND (
                $exam_name = ""
                OR searchable_exam CONTAINS $exam_name
                OR searchable_exam_type CONTAINS $exam_name
            )
            AND (
                $severity = ""
                OR toLower(coalesce(s.severity_normalized, "")) CONTAINS $severity
            )
            AND (
                $clinical_history = ""
                OR searchable_history CONTAINS $clinical_history
                OR searchable_finding_text CONTAINS $clinical_history
            )
            AND (
                $modality = ""
                OR searchable_modality_name CONTAINS $modality
                OR searchable_modality_type CONTAINS $modality
                OR searchable_exam CONTAINS $modality
            )
            AND (
                $organ = ""
                OR any(x IN organs_norm WHERE x CONTAINS $organ)
                OR searchable_finding_text CONTAINS $organ
                OR toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS $organ
            )
            AND (
                $finding = ""
                OR any(x IN finding_terms_norm WHERE x CONTAINS $finding)
                OR searchable_finding_text CONTAINS $finding
                OR toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS $finding
            )
            AND (
                $min_dimension_cm IS NULL
                OR (max_dimension_cm IS NOT NULL AND max_dimension_cm >= $min_dimension_cm)
                OR toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS toString($min_dimension_cm)
            )
            AND (
                $max_dimension_cm IS NULL
                OR (max_dimension_cm IS NOT NULL AND max_dimension_cm <= $max_dimension_cm)
            )

        WITH p, s, e, m, f, organs, finding_terms, measurement_texts, max_dimension_cm, searchable_finding_text,
             reduce(keyword_score = 0.0, kw IN $keywords |
                keyword_score +
                CASE WHEN kw = "" THEN 0.0
                     WHEN searchable_finding_text CONTAINS kw THEN 2.5
                     ELSE 0.0 END +
                CASE WHEN kw = "" THEN 0.0
                     WHEN toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS kw THEN 2.0
                     ELSE 0.0 END +
                CASE WHEN kw = "" THEN 0.0
                     WHEN any(x IN organs WHERE toLower(x) CONTAINS kw) THEN 1.5
                     ELSE 0.0 END +
                CASE WHEN kw = "" THEN 0.0
                     WHEN any(x IN finding_terms WHERE toLower(x) CONTAINS kw) THEN 1.5
                     ELSE 0.0 END +
                CASE WHEN kw = "" THEN 0.0
                     WHEN toLower(coalesce(e.name_normalized, "")) CONTAINS kw THEN 1.0
                     ELSE 0.0 END
             ) AS keyword_score

        WITH p, s, e, m, f, organs, finding_terms, measurement_texts, max_dimension_cm, keyword_score,
             (
                CASE WHEN $gender <> "" AND toLower(coalesce(p.gender, "")) = $gender THEN 1.5 ELSE 0.0 END +
                CASE WHEN $min_age IS NOT NULL OR $max_age IS NOT NULL THEN 1.0 ELSE 0.0 END +
                CASE WHEN $modality <> "" AND (
                    toLower(coalesce(m.name_normalized, "")) CONTAINS $modality OR
                    toLower(coalesce(m.type_normalized, "")) CONTAINS $modality OR
                    toLower(coalesce(e.name_normalized, "")) CONTAINS $modality
                ) THEN 2.0 ELSE 0.0 END +
                CASE WHEN $organ <> "" AND (
                    any(x IN organs WHERE toLower(x) CONTAINS $organ) OR
                    toLower(coalesce(f.text_normalized, "")) CONTAINS $organ OR
                    toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS $organ
                ) THEN 3.0 ELSE 0.0 END +
                CASE WHEN $finding <> "" AND (
                    any(x IN finding_terms WHERE toLower(x) CONTAINS $finding) OR
                    toLower(coalesce(f.text_normalized, "")) CONTAINS $finding OR
                    toLower(coalesce(s.result_text_plain_normalized, "")) CONTAINS $finding
                ) THEN 3.0 ELSE 0.0 END +
                CASE WHEN $min_dimension_cm IS NOT NULL AND max_dimension_cm IS NOT NULL AND max_dimension_cm >= $min_dimension_cm THEN 2.0 ELSE 0.0 END +
                CASE WHEN $max_dimension_cm IS NOT NULL AND max_dimension_cm IS NOT NULL AND max_dimension_cm <= $max_dimension_cm THEN 1.5 ELSE 0.0 END +
                CASE WHEN $exam_name <> "" AND (
                    toLower(coalesce(e.name_normalized, "")) CONTAINS $exam_name OR
                    toLower(coalesce(e.type_normalized, "")) CONTAINS $exam_name
                ) THEN 2.0 ELSE 0.0 END +
                CASE WHEN $severity <> "" AND toLower(coalesce(s.severity_normalized, "")) CONTAINS $severity THEN 1.0 ELSE 0.0 END +
                CASE WHEN $clinical_history <> "" AND toLower(coalesce(s.clinical_instruction_normalized, "")) CONTAINS $clinical_history THEN 1.0 ELSE 0.0 END
             ) AS structured_score

        RETURN
            'graph' AS retrieval_mode,
            p.id AS hn,
            p.gender AS gender,
            p.age AS age,
            s.id AS study_id,
            s.severity AS severity,
            s.clinical_instruction AS clinical_instruction,
            s.result_text_plain AS result_text_plain,
            coalesce(m.name, m.type) AS modality,
            e.name AS exam_name,
            coalesce(f.id, s.id + '_study_fallback') AS finding_id,
            coalesce(f.section, 'study_level') AS section,
            coalesce(f.text, s.result_text_plain) AS finding_text,
            organs AS organs,
            finding_terms AS finding_terms,
            measurement_texts AS measurement_texts,
            max_dimension_cm AS max_dimension_cm,
            round((structured_score + keyword_score) * 100.0) / 100.0 AS score
        ORDER BY score DESC, study_id ASC, finding_id ASC
        LIMIT $top_k
        """

        try:
            with self.gb.driver.session() as session:
                rows = session.run(graph_query, **params)
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _vector_search(self, raw_query: str, params: dict, top_k: int) -> List[Dict[str, Any]]:
        if not VECTOR_ENABLED or not self.eg.is_ready():
            return []

        try:
            embeddings = self.eg.generate_embeddings([raw_query])
            if not embeddings:
                return []
            query_embedding = embeddings[0]
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        vector_query = """
        CALL db.index.vector.queryNodes('finding_text_embeddings', $top_k, $embedding)
        YIELD node, score

        MATCH (s:Study)-[:HAS_FINDING]->(node)
        MATCH (p:Patient)-[:HAS_STUDY]->(s)
        OPTIONAL MATCH (s)-[:HAS_EXAM]->(e:Exam)
        OPTIONAL MATCH (s)-[:HAS_MODALITY]->(m:Modality)
        OPTIONAL MATCH (node)-[:LOCATED_IN]->(o:Organ)
        OPTIONAL MATCH (node)-[:HAS_FINDING_TERM]->(ft:FindingTerm)
        OPTIONAL MATCH (node)-[:HAS_MEASUREMENT]->(meas:Measurement)

        WITH p, s, node AS f, e, m, score,
             collect(DISTINCT o.name) AS organs,
             collect(DISTINCT o.name_normalized) AS organs_norm,
             collect(DISTINCT ft.name) AS finding_terms,
             collect(DISTINCT ft.name_normalized) AS finding_terms_norm,
             collect(DISTINCT meas.raw_text) AS measurement_texts,
             max(meas.max_dimension_cm) AS max_dimension_cm

        WHERE
            ($gender = "" OR toLower(coalesce(p.gender, "")) = $gender)
            AND ($min_age IS NULL OR toInteger(coalesce(p.age, 0)) >= $min_age)
            AND ($max_age IS NULL OR toInteger(coalesce(p.age, 0)) <= $max_age)
            AND ($accession_no = "" OR s.id = $accession_no)
            AND ($exam_name = "" OR toLower(coalesce(e.name_normalized, "")) CONTAINS $exam_name OR toLower(coalesce(e.type_normalized, "")) CONTAINS $exam_name)
            AND ($severity = "" OR toLower(coalesce(s.severity_normalized, "")) CONTAINS $severity)
            AND ($clinical_history = "" OR toLower(coalesce(s.clinical_instruction_normalized, "")) CONTAINS $clinical_history)
            AND ($modality = "" OR toLower(coalesce(m.name_normalized, "")) CONTAINS $modality OR toLower(coalesce(m.type_normalized, "")) CONTAINS $modality OR toLower(coalesce(e.name_normalized, "")) CONTAINS $modality)
            AND ($organ = "" OR any(x IN organs_norm WHERE x CONTAINS $organ) OR toLower(coalesce(f.text_normalized, "")) CONTAINS $organ)
            AND ($finding = "" OR any(x IN finding_terms_norm WHERE x CONTAINS $finding) OR toLower(coalesce(f.text_normalized, "")) CONTAINS $finding)
            AND ($min_dimension_cm IS NULL OR (max_dimension_cm IS NOT NULL AND max_dimension_cm >= $min_dimension_cm))
            AND ($max_dimension_cm IS NULL OR (max_dimension_cm IS NOT NULL AND max_dimension_cm <= $max_dimension_cm))

        RETURN
            'vector' AS retrieval_mode,
            p.id AS hn,
            p.gender AS gender,
            p.age AS age,
            s.id AS study_id,
            s.severity AS severity,
            s.clinical_instruction AS clinical_instruction,
            s.result_text_plain AS result_text_plain,
            coalesce(m.name, m.type) AS modality,
            e.name AS exam_name,
            f.id AS finding_id,
            f.section AS section,
            f.text AS finding_text,
            organs AS organs,
            finding_terms AS finding_terms,
            measurement_texts AS measurement_texts,
            max_dimension_cm AS max_dimension_cm,
            round(score * 1000.0) / 1000.0 AS score
        ORDER BY score DESC, study_id ASC, finding_id ASC
        LIMIT $top_k
        """

        vector_params = dict(params)
        vector_params["embedding"] = query_embedding
        vector_params["top_k"] = int(max(top_k, 10))

        try:
            with self.gb.driver.session() as session:
                rows = session.run(vector_query, **vector_params)
                return [dict(r) for r in rows][:top_k]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def search(self, parsed_query: dict, raw_query: str, top_k: int = DEFAULT_TOP_K, advanced_filters: dict = None) -> List[Dict[str, Any]]:
        merged = self._merge_filters(parsed_query, advanced_filters)
        params = self._base_params(merged, raw_query, top_k)
        logger.info(f"Search params: {format_json(params)}")

        graph_results = self._graph_search(params)
        if graph_results:
            return graph_results

        logger.warning("Graph search empty; trying vector search")
        vector_results = self._vector_search(raw_query, params, top_k)
        return vector_results


# =========================================================
# REPORT BUILD / MISSING ORGANS
# =========================================================
class ClinicalContextBuilder:
    @staticmethod
    def infer_study_bucket(results: List[Dict[str, Any]], query: str = "") -> Optional[str]:
        for r in results:
            bucket = modality_to_study_bucket(r.get("modality", ""), r.get("exam_name", ""))
            if bucket:
                return bucket
        return modality_to_study_bucket("", query)

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        organs = []
        findings = []
        measurements = []
        sections = []
        exam_names = []
        modalities = []

        for r in results:
            organs.extend(r.get("organs") or [])
            findings.extend(r.get("finding_terms") or [])
            measurements.extend(r.get("measurement_texts") or [])
            if r.get("section"):
                sections.append(r["section"])
            if r.get("exam_name"):
                exam_names.append(r["exam_name"])
            if r.get("modality"):
                modalities.append(r["modality"])

        return {
            "organs": unique_preserve_order(organs),
            "finding_terms": unique_preserve_order(findings),
            "measurement_texts": unique_preserve_order(measurements),
            "sections": unique_preserve_order(sections),
            "exam_names": unique_preserve_order(exam_names),
            "modalities": unique_preserve_order(modalities),
        }

    @staticmethod
    def generate_missing_organ_lines(results: List[Dict[str, Any]], query: str = "") -> List[str]:
        bucket = ClinicalContextBuilder.infer_study_bucket(results, query)
        if not bucket or bucket not in STUDY_TEMPLATES:
            return []

        expected = STUDY_TEMPLATES[bucket]["organs"]
        default_text = STUDY_TEMPLATES[bucket]["default_text"]

        present = set()
        for r in results:
            for organ in r.get("organs") or []:
                present.add(normalize_text(organ))
            finding_text = normalize_text(r.get("finding_text", ""))
            for organ in expected:
                if normalize_text(organ) in finding_text:
                    present.add(normalize_text(organ))

        missing = [org for org in expected if normalize_text(org) not in present]
        return [f"{org.title()}: {default_text}" for org in missing]


# =========================================================
# GENERATION
# =========================================================
class ReportGenerator:
    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.url = url
        self.model = model

    def is_ready(self) -> Tuple[bool, str]:
        try:
            payload = {"model": self.model, "prompt": "reply with OK", "stream": False}
            response = requests.post(f"{self.url}/api/generate", json=payload, timeout=15)
            response.raise_for_status()
            return True, "Ollama connected"
        except Exception as e:
            return False, f"Ollama error: {e}"

    def _post(self, payload: dict, timeout: int = 120) -> dict:
        response = requests.post(f"{self.url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def build_context_for_generation(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        compact_rows = []
        for idx, r in enumerate(search_results[:MAX_CONTEXT_RESULTS], start=1):
            compact_rows.append({
                "rank": idx,
                "study_id": r.get("study_id"),
                "retrieval_mode": r.get("retrieval_mode"),
                "age": r.get("age"),
                "gender": r.get("gender"),
                "modality": r.get("modality"),
                "exam_name": r.get("exam_name"),
                "section": r.get("section"),
                "organs": r.get("organs"),
                "finding_terms": r.get("finding_terms"),
                "measurement_texts": r.get("measurement_texts"),
                "max_dimension_cm": r.get("max_dimension_cm"),
                "finding_text": r.get("finding_text"),
                "clinical_instruction": r.get("clinical_instruction"),
            })

        summary = ClinicalContextBuilder.summarize_results(search_results)
        missing_lines = ClinicalContextBuilder.generate_missing_organ_lines(search_results, query)

        payload = {
            "user_query": query,
            "retrieved_evidence": compact_rows,
            "summary": summary,
            "missing_organs_auto_generated": missing_lines
        }
        return format_json(payload)

    def generate_report(self, query: str, search_results: List[Dict[str, Any]], report_type: str, system_prompt: str) -> str:
        if not search_results:
            return "No evidence found. Please relax filters or ingest data first."

        context = self.build_context_for_generation(query, search_results)

        output_hint = {
            "Full Report": "Return Findings, Impression, and Recommendations.",
            "Only Findings": "Return only the Findings section.",
            "Only Impression": "Return only Impression section in numbered form.",
            "Recommendations": "Return only Recommendations."
        }.get(report_type, "Return a complete professional report.")

        prompt = f"""
{system_prompt}

You are generating a production radiology answer using retrieved clinical evidence.
Rules:
- Use only clinically supported details from evidence.
- You may include auto-generated missing organ statements from the supplied payload.
- Do not mention 'context', 'payload', 'retrieval', or 'query'.
- If evidence is limited, stay conservative and explicit.
- Preserve organs, side, measurements, modality cues, and severity if present.
- If a study template indicates missing organs, integrate them naturally.

Requested format:
{output_hint}

Evidence payload:
{context}
""".strip()

        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            return self._post(payload, timeout=120).get("response", "").strip()
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            fallback_lines = ClinicalContextBuilder.generate_missing_organ_lines(search_results, query)
            rows = []
            for r in search_results[:5]:
                rows.append(
                    f"- {r.get('exam_name') or r.get('modality')}: {truncate_text(r.get('finding_text', ''), 220)}"
                )
            return "\n".join([
                "LLM generation unavailable. Deterministic evidence summary:",
                *rows,
                "",
                "Auto-added missing organ statements:",
                *(fallback_lines or ["- None"])
            ])

    def answer_question(self, user_question: str, context_payload: str, exam_filter: str = "") -> str:
        prompt = f"""
You are a radiology QA assistant.
Answer the user's question using the supplied clinical evidence only.
Be concise, accurate, and explicit when evidence is absent.
Do not hallucinate.

Exam filter: {exam_filter}

Evidence:
{context_payload}

Question:
{user_question}
""".strip()
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            return self._post(payload, timeout=60).get("response", "").strip()
        except Exception as e:
            logger.error(f"QA failed: {e}")
            return "I could not reach the language model. Please check Ollama service."


# =========================================================
# INGESTION
# =========================================================
def ingest_pipeline(data_path: str):
    logger.info("Starting ingestion pipeline")
    dl = DataLoader(data_path)
    rp = ReportParser()
    ee = EntityExtractor()
    gb = GraphBuilder()
    eg = EmbeddingGenerator()
    vib = VectorIndexBuilder(gb)

    records = dl.load_data()
    gb.setup_constraints()
    gb.setup_indexes()

    if eg.is_ready():
        vib.create_finding_vector_index(vector_dimension=eg.embedding_dimension)

    total = len(records)
    for idx, record in enumerate(records, start=1):
        text = record.get("RESULT_TEXT_PLAIN", "")
        sections = rp.parse_sections(text)
        entities = ee.process_sections(sections)
        gb.ingest_report(record, entities)

        if eg.is_ready():
            study_id = record.get("ACCESSION_NO") or ""
            for sec, entity_list in entities.items():
                for i, ent in enumerate(entity_list):
                    text_block = ent.get("context_block", "")
                    if text_block:
                        try:
                            vecs = eg.generate_embeddings([text_block])
                            if vecs:
                                vib.update_finding_embeddings(f"{study_id}_{sec}_{i}", vecs[0])
                        except Exception as e:
                            logger.warning(f"Embedding update failed for {study_id}/{sec}/{i}: {e}")

        if idx % 10 == 0 or idx == total:
            logger.info(f"Ingested {idx}/{total}")

    gb.close()
    logger.info("Ingestion complete")


# =========================================================
# UI HELPERS
# =========================================================
def render_status(gb: GraphBuilder, rg: ReportGenerator, eg: EmbeddingGenerator):
    neo_ok, neo_msg = gb.verify()
    llm_ok, llm_msg = rg.is_ready()
    emb_ok = eg.is_ready()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Neo4j", "Connected" if neo_ok else "Down")
        st.caption(neo_msg)
    with col2:
        st.metric("Ollama", "Connected" if llm_ok else "Down")
        st.caption(llm_msg)
    with col3:
        st.metric("Embeddings", "Ready" if emb_ok else "Disabled")
        st.caption(eg.model_name if emb_ok else "SentenceTransformer unavailable")


def build_advanced_filters_from_sidebar() -> Dict[str, Any]:
    return {
        "exam_name": st.session_state.get("adv_exam_name", ""),
        "accession_no": st.session_state.get("adv_acc", ""),
        "severity": st.session_state.get("adv_sev", ""),
        "clinical_history": st.session_state.get("adv_history", ""),
        "modality": st.session_state.get("adv_mod", ""),
        "organ_name": st.session_state.get("adv_organ", ""),
        "finding": st.session_state.get("adv_finding", ""),
        "min_age": st.session_state.get("adv_min_age"),
        "max_age": st.session_state.get("adv_max_age"),
        "min_dimension_cm": parse_size_to_cm(st.session_state.get("adv_min_dim", "")),
        "max_dimension_cm": parse_size_to_cm(st.session_state.get("adv_max_dim", "")),
    }


# =========================================================
# STREAMLIT APP
# =========================================================
def run_streamlit_app():
    st.set_page_config(page_title="Production Radiology System", layout="wide")

    @st.cache_resource
    def init_services():
        gb = GraphBuilder()
        eg = EmbeddingGenerator()
        qp = QueryParserLLM()
        hs = HybridSearch(gb, eg)
        rg = ReportGenerator()
        return gb, eg, qp, hs, rg

    gb, eg, qp, hs, rg = init_services()

    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "parsed_query" not in st.session_state:
        st.session_state.parsed_query = {}
    if "current_context_payload" not in st.session_state:
        st.session_state.current_context_payload = ""
    if "last_report" not in st.session_state:
        st.session_state.last_report = ""
    if "filter_prefix" not in st.session_state:
        st.session_state.filter_prefix = ""

    st.title(APP_TITLE)
    render_status(gb, rg, eg)

    with st.sidebar:
        st.header("🔍 Left Menu Filters")
        st.caption("Autocomplete-backed filters from database values")

        st.text_input(
            "Filter suggestion search",
            key="filter_prefix",
            placeholder="Type to narrow dropdown options"
        )

        options = gb.get_filter_options(st.session_state.filter_prefix, limit=120)

        st.text_input("Accession No", key="adv_acc", placeholder="Exact accession if known")

        col_age1, col_age2 = st.columns(2)
        with col_age1:
            st.number_input("Min Age", min_value=0, max_value=120, value=0, key="adv_min_age")
        with col_age2:
            st.number_input("Max Age", min_value=0, max_value=120, value=120, key="adv_max_age")

        st.selectbox(
            "Exam Name",
            options=[""] + options.get("exams", []),
            key="adv_exam_name",
            index=0,
            help="Autocomplete-like dropdown from Neo4j"
        )
        st.selectbox(
            "Organ Name",
            options=[""] + (options.get("organs", []) or KNOWN_ORGANS),
            key="adv_organ",
            index=0
        )
        st.selectbox(
            "Finding Term",
            options=[""] + (options.get("findings", []) or KNOWN_FINDINGS),
            key="adv_finding",
            index=0
        )
        st.selectbox(
            "Modality",
            options=[""] + options.get("modalities", []),
            key="adv_mod",
            index=0
        )
        st.selectbox(
            "Severity",
            options=[""] + options.get("severity", []),
            key="adv_sev",
            index=0
        )
        st.selectbox(
            "Clinical History",
            options=[""] + options.get("history", []),
            key="adv_history",
            index=0
        )

        st.text_input("Min lesion size", key="adv_min_dim", placeholder="e.g. 1 cm or 10 mm")
        st.text_input("Max lesion size", key="adv_max_dim", placeholder="e.g. 2 cm or 20 mm")

        if st.button("Clear Filters"):
            for k in ["adv_acc", "adv_exam_name", "adv_organ", "adv_finding", "adv_mod", "adv_sev", "adv_history", "adv_min_dim", "adv_max_dim"]:
                st.session_state[k] = ""
            st.session_state["adv_min_age"] = 0
            st.session_state["adv_max_age"] = 120
            st.rerun()

    top_query_col, top_action_col = st.columns([4, 1])

    with top_query_col:
        query = st.text_input(
            "Ask or search",
            placeholder="e.g. Show female patients over 60 with thyroid nodule more than 1 cm on US"
        )
    with top_action_col:
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=DEFAULT_TOP_K)

    with st.expander("Generation settings", expanded=True):
        colp1, colp2 = st.columns([2, 1])
        with colp1:
            system_prompt_input = st.text_area(
                "Radiology System Prompt",
                value="""You are a Board-Certified Diagnostic Radiologist.
Generate a professional radiology answer.

Rules:
1. Preserve every organ, side, and measurement supported by evidence.
2. Use conservative wording when evidence is limited.
3. If the supplied evidence contains auto-generated missing organ lines for the inferred study, integrate them naturally.
4. Do not mention internal system behavior.
5. Prefer formal radiology phrasing and structured output.""",
                height=180
            )
        with colp2:
            report_focus = st.radio(
                "Output Type",
                ["Full Report", "Only Findings", "Only Impression", "Recommendations"],
                index=0
            )

    advanced_filters = build_advanced_filters_from_sidebar()

    if st.button("🚀 Run Search", type="primary", use_container_width=True):
        with st.spinner("Searching database and fallback retrieval..."):
            parsed_params = qp.parse_query(query) if query.strip() else {}
            st.session_state.parsed_query = parsed_params
            st.session_state.search_results = hs.search(
                parsed_query=parsed_params,
                raw_query=query,
                top_k=int(top_k),
                advanced_filters=advanced_filters
            )
            st.session_state.chat_history = []
            st.session_state.last_report = ""

            if st.session_state.search_results:
                st.session_state.current_context_payload = rg.build_context_for_generation(
                    query, st.session_state.search_results
                )
            else:
                st.session_state.current_context_payload = ""

    tab_query, tab_ingest, tab_debug = st.tabs(["Diagnostic Query & QA", "Data Ingestion", "Debug"])

    with tab_ingest:
        st.subheader("Graph Knowledge Ingestion")
        ingest_path = st.text_input(
            "Absolute path to JSON file or directory",
            placeholder=r"d:\Anigravity_2\rows_00001-01000.json"
        )
        if st.button("Start Ingestion"):
            if not ingest_path:
                st.warning("Provide a valid path")
            else:
                with st.spinner("Ingesting records into Neo4j..."):
                    try:
                        ingest_pipeline(ingest_path)
                        st.success(f"Ingestion successful: {ingest_path}")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    with tab_query:
        if st.session_state.parsed_query:
            with st.expander("Parsed Query", expanded=False):
                st.json(st.session_state.parsed_query)

        if not st.session_state.search_results:
            st.info("No matching records found. Try relaxing filters, confirming ingestion, or checking debug diagnostics.")
        else:
            st.success(f"Retrieved {len(st.session_state.search_results)} evidence blocks")

            summary = ClinicalContextBuilder.summarize_results(st.session_state.search_results)
            missing_lines = ClinicalContextBuilder.generate_missing_organ_lines(st.session_state.search_results, query)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Results", len(st.session_state.search_results))
            metric_cols[1].metric("Unique organs", len(summary["organs"]))
            metric_cols[2].metric("Unique findings", len(summary["finding_terms"]))
            metric_cols[3].metric("Auto-missing organs", len(missing_lines))

            right_col1, right_col2 = st.columns([1, 1])

            with right_col1:
                st.subheader("Source Data")
                df = pd.DataFrame(st.session_state.search_results)
                if not df.empty:
                    cols = [
                        "retrieval_mode", "study_id", "hn", "age", "gender", "modality", "exam_name",
                        "severity", "score", "section", "organs", "finding_terms",
                        "measurement_texts", "max_dimension_cm", "finding_text"
                    ]
                    ordered_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
                    st.dataframe(df[ordered_cols], use_container_width=True, height=420)

                with st.expander("Raw Context Payload", expanded=False):
                    st.code(st.session_state.current_context_payload, language="json")

            with right_col2:
                st.subheader("Generated Report / Answer")
                if st.button("✨ Generate Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        st.session_state.last_report = rg.generate_report(
                            query=query,
                            search_results=st.session_state.search_results,
                            report_type=report_focus,
                            system_prompt=system_prompt_input
                        )

                if st.session_state.last_report:
                    st.text_area("Output", value=st.session_state.last_report, height=420)

                if missing_lines:
                    with st.expander("Auto-generated Missing Organ Coverage", expanded=False):
                        for line in missing_lines:
                            st.markdown(f"- {line}")

            st.subheader("Interactive Question Answering")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_msg := st.chat_input("Ask a follow-up question about the retrieved cases"):
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)

                with st.chat_message("assistant"):
                    with st.spinner("Answering..."):
                        ans = rg.answer_question(
                            user_question=user_msg,
                            context_payload=st.session_state.current_context_payload,
                            exam_filter=advanced_filters.get("exam_name", "")
                        )
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})

    with tab_debug:
        st.subheader("Diagnostics")
        neo_ok, neo_msg = gb.verify()
        llm_ok, llm_msg = rg.is_ready()

        st.write({
            "neo4j": neo_msg,
            "ollama": llm_msg,
            "embeddings_ready": eg.is_ready(),
            "vector_enabled": VECTOR_ENABLED,
            "advanced_filters": advanced_filters
        })

        if st.button("Run DB Count Checks"):
            try:
                counts = {}
                qmap = {
                    "patients": "MATCH (n:Patient) RETURN count(n) AS c",
                    "studies": "MATCH (n:Study) RETURN count(n) AS c",
                    "findings": "MATCH (n:Finding) RETURN count(n) AS c",
                    "organs": "MATCH (n:Organ) RETURN count(n) AS c",
                    "finding_terms": "MATCH (n:FindingTerm) RETURN count(n) AS c",
                    "measurements": "MATCH (n:Measurement) RETURN count(n) AS c",
                }
                with gb.driver.session() as session:
                    for k, q in qmap.items():
                        counts[k] = session.run(q).single()["c"]
                st.json(counts)
            except Exception as e:
                st.error(f"Count query failed: {e}")

        if st.button("Run Sample Search Smoke Test"):
            try:
                smoke = hs.search({}, "thyroid nodule", top_k=5, advanced_filters={})
                st.json(smoke)
            except Exception as e:
                st.error(f"Smoke test failed: {e}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    if "streamlit" in sys.modules or "streamlit" in sys.argv[0]:
        run_streamlit_app()
    else:
        parser = argparse.ArgumentParser(description="Production Radiology QA/Search System")
        parser.add_argument("--ingest", type=str, help="Path to JSON file or directory to ingest")
        parser.add_argument("--verify", action="store_true", help="Verify Neo4j and Ollama connectivity")
        args = parser.parse_args()

        if args.verify:
            gb = GraphBuilder()
            rg = ReportGenerator()
            eg = EmbeddingGenerator()
            print({"neo4j": gb.verify(), "ollama": rg.is_ready(), "embeddings": eg.is_ready()})
            gb.close()
        elif args.ingest:
            ingest_pipeline(args.ingest)
        else:
            parser.print_help()
