"""Controlled relation vocabulary, entity alias normalization, and provenance tracking.

Provides standardized entity resolution and canonical predicate mapping for the
ISRO domain knowledge graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProvenanceTriple:
    """Explicit knowledge graph triple with provenance linkage."""
    subject: str
    relation: str
    object: str
    document_id: str = ""
    chunk_id: str = ""
    source_url: str = ""
    confidence: float = 1.0

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def to_natural_language(self) -> str:
        readable_rel = self.relation.lower().replace("_", " ")
        return f"{self.subject} {readable_rel} {self.object}."


# Canonical entity alias mapping for aerospace and ISRO nomenclature
ENTITY_ALIASES: dict[str, str] = {
    # Missions
    "aditya l1": "Aditya-L1",
    "aditya-l1 mission": "Aditya-L1",
    "aditya l-1": "Aditya-L1",
    "aditya-1": "Aditya-L1",
    "chandrayaan 1": "Chandrayaan-1",
    "chandrayaan-1 mission": "Chandrayaan-1",
    "chandrayaan 2": "Chandrayaan-2",
    "chandrayaan-2 mission": "Chandrayaan-2",
    "chandrayaan-ii": "Chandrayaan-2",
    "chandrayaan 3": "Chandrayaan-3",
    "chandrayaan-3 mission": "Chandrayaan-3",
    "chandrayaan-iii": "Chandrayaan-3",
    "mangalyaan": "Mars Orbiter Mission",
    "mom": "Mars Orbiter Mission",
    "mars orbiter mission": "Mars Orbiter Mission",
    "gaganyaan": "Gaganyaan",
    "gaganyaan mission": "Gaganyaan",
    "astrosat": "AstroSat",
    "risat-2br1": "RISAT-2BR1",
    "cartosat-3": "Cartosat-3",
    "xposat": "XPoSat",

    # Launch Vehicles
    "pslv": "PSLV",
    "polar satellite launch vehicle": "PSLV",
    "pslv c37": "PSLV-C37",
    "pslv-c37": "PSLV-C37",
    "pslv-c57": "PSLV-C57",
    "gslv": "GSLV",
    "geosynchronous satellite launch vehicle": "GSLV",
    "gslv mk iii": "LVM3 / GSLV Mk III",
    "gslv mk-iii": "LVM3 / GSLV Mk III",
    "gslv mk3": "LVM3 / GSLV Mk III",
    "lvm3": "LVM3 / GSLV Mk III",
    "sslv": "SSLV",

    # Payloads
    "velc": "VELC",
    "visible emission line coronagraph": "VELC",
    "suit": "SUIT",
    "solar ultraviolet imaging telescope": "SUIT",
    "aspex": "ASPEX",
    "aditya solar wind particle experiment": "ASPEX",
    "papa": "PAPA",
    "plasma analyser package for aditya": "PAPA",
    "solexs": "SoLEXS",
    "solar low energy x-ray spectrometer": "SoLEXS",
    "hel1os": "HEL1OS",
    "high energy l1 orbiting x-ray spectrometer": "HEL1OS",
    "mag": "MAG",
    "magnetometer": "MAG",

    # Organizations & Centers
    "isro": "ISRO",
    "indian space research organisation": "ISRO",
    "indian space research organization": "ISRO",
    "issdc": "ISSDC",
    "indian space science data centre": "ISSDC",
    "ursc": "URSC",
    "u r rao satellite centre": "URSC",
    "isac": "URSC",
    "vssc": "VSSC",
    "vikram sarabhai space centre": "VSSC",
    "sdsc": "SDSC SHAR",
    "sdsc shar": "SDSC SHAR",
    "satish dhawan space centre": "SDSC SHAR",
    "shar": "SDSC SHAR",
    "sac": "SAC",
    "space applications centre": "SAC",
    "iist": "IIST",
    "indian institute of space science and technology": "IIST",
    "dos": "Department of Space",
    "department of space": "Department of Space",
    "iiap": "IIA",
    "indian institute of astrophysics": "IIA",
    "iia": "IIA",
    "iucaa": "IUCAA",
}

# Controlled relation vocabulary mapping
RAW_RELATION_MAP: dict[str, str] = {
    # Payloads & equipment
    "carry": "HAS_PAYLOAD",
    "carries": "HAS_PAYLOAD",
    "equipped": "HAS_PAYLOAD",
    "contain": "HAS_PAYLOAD",
    "include": "HAS_PAYLOAD",
    "has_payload": "HAS_PAYLOAD",
    "payload": "HAS_PAYLOAD",

    # Launch events
    "launch": "LAUNCHED_BY",
    "launched_by": "LAUNCHED_BY",
    "launch_by": "LAUNCHED_BY",
    "launched_from": "LAUNCHED_FROM",
    "lift": "LAUNCHED_BY",
    "launched_on": "LAUNCHED_ON",
    "launch_on": "LAUNCHED_ON",
    "launch_date": "LAUNCHED_ON",

    # Development & agency
    "develop": "DEVELOPED_BY",
    "designed_by": "DEVELOPED_BY",
    "build": "DEVELOPED_BY",
    "built_by": "DEVELOPED_BY",
    "fabricate": "DEVELOPED_BY",
    "created_by": "DEVELOPED_BY",
    "lead": "DEVELOPED_BY",
    "operate": "OPERATED_BY",
    "managed_by": "OPERATED_BY",

    # Trajectory & destination
    "orbit": "ORBITS_AT",
    "reach": "REACHED",
    "arrive": "REACHED",
    "land": "LANDED_ON",
    "insert": "INSERTED_INTO",
    "placed_in": "OPERATES_AT",
    "operates_at": "OPERATES_AT",

    # Scientific objectives & observations
    "observe": "OBSERVES",
    "study": "OBSERVES",
    "measure": "MEASURES",
    "detect": "MEASURES",
    "image": "OBSERVES",
    "investigate": "HAS_OBJECTIVE",
    "explore": "HAS_OBJECTIVE",
    "monitor": "MONITORS",
    "has_objective": "HAS_OBJECTIVE",

    # Composition & hierarchy
    "part_of": "PART_OF",
    "consist": "HAS_PART",
    "comprise": "HAS_PART",
    "has_part": "HAS_PART",
    "collaborate": "COLLABORATES_WITH",
    "partner": "COLLABORATES_WITH",
}


def normalize_entity(name: str) -> str:
    """Normalize entity text to its canonical domain label."""
    if not name or not name.strip():
        return ""
    cleaned = re.sub(r"\s+", " ", name.strip())
    # Strip leading/trailing articles or punctuation
    cleaned_lower = re.sub(r"^(?:the|a|an)\s+", "", cleaned.lower()).strip(".,;:()")
    return ENTITY_ALIASES.get(cleaned_lower, cleaned)


def map_relation(raw_rel: str) -> str:
    """Map a raw dependency/verb predicate to the controlled vocabulary."""
    if not raw_rel:
        return "RELATED_TO"
    cleaned = raw_rel.strip().lower().replace(" ", "_")
    return RAW_RELATION_MAP.get(cleaned, cleaned.upper() if len(cleaned) <= 15 else "RELATED_TO")


def create_provenance_triple(
    subject: str,
    relation: str,
    object_: str,
    document_id: str = "",
    chunk_id: str = "",
    source_url: str = "",
    confidence: float = 1.0,
) -> ProvenanceTriple:
    """Create a canonical normalized triple with provenance metadata."""
    norm_subj = normalize_entity(subject)
    norm_obj = normalize_entity(object_)
    norm_rel = map_relation(relation)
    return ProvenanceTriple(
        subject=norm_subj,
        relation=norm_rel,
        object=norm_obj,
        document_id=document_id,
        chunk_id=chunk_id,
        source_url=source_url,
        confidence=confidence,
    )
