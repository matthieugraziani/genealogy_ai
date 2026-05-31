"""
Extraction d'entités depuis le texte OCR.
Stratégie en deux passes :
  1. Regex rapide (déterministe, pas besoin de GPU)
  2. LLM structuré (Mistral / Llama via Ollama) pour les cas complexes
"""

from __future__ import annotations
import json
import re
from typing import Optional

from .entities import Person, FamilyRelation, ExtractionResult


# ───────────────────────────── Patterns regex ─────────────────────────────

_MOIS = (
    r"janvier|février|mars|avril|mai|juin|juillet|août|septembre"
    r"|octobre|novembre|décembre"
)

PATTERNS = {
    "naissance": re.compile(
        r"(?:le\s+)?(\d{1,2})\s+(" + _MOIS + r")\s+(\d{4})\s+est\s+né[e]?\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+)",
        re.IGNORECASE,
    ),
    "fils_de": re.compile(
        r"fils\s+de\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+)\s+et\s+(?:de\s+)?([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+)",
        re.IGNORECASE,
    ),
    "fille_de": re.compile(
        r"fille\s+de\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+)\s+et\s+(?:de\s+)?([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+)",
        re.IGNORECASE,
    ),
    "mariage": re.compile(
        r"(?:le\s+)?(\d{1,2})\s+(" + _MOIS + r")\s+(\d{4}).*?mariage.*?([A-ZÀ-Ÿ][a-zà-ÿ\s]+?)\s+(?:et|avec)\s+([A-ZÀ-Ÿ][a-zà-ÿ\s]+)",
        re.IGNORECASE | re.DOTALL,
    ),
    "lieu": re.compile(r"à\s+([A-ZÀ-Ÿ][a-zà-ÿ\-]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ\-]+)*)", re.IGNORECASE),
}


def _split_nom_prenom(full_name: str) -> tuple[str, str]:
    """Heuristique simple : dernier mot = NOM, reste = prénom(s)."""
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return parts[-1].upper(), " ".join(parts[:-1]).title()
    return full_name.upper(), ""


class RegexExtractor:
    """Extraction rapide basée sur des patterns pour actes français XIXe-XXe."""

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        result = ExtractionResult(document_path=source, raw_text=text)

        # --- Détecter le type de document ---
        if re.search(r"\bné[e]?\b", text, re.IGNORECASE):
            result.document_type = "naissance"
        elif re.search(r"\bmariage\b|\bmariés\b", text, re.IGNORECASE):
            result.document_type = "mariage"
        elif re.search(r"\bdécédé[e]?\b|\bdécès\b", text, re.IGNORECASE):
            result.document_type = "deces"

        # --- Naissance ---
        m = PATTERNS["naissance"].search(text)
        if m:
            jour, mois, annee, full_name = m.group(1), m.group(2), m.group(3), m.group(4)
            nom, prenom = _split_nom_prenom(full_name)
            enfant = Person(
                nom=nom,
                prenom=prenom,
                date_naissance=f"{jour} {mois} {annee}",
                source_document=source,
            )
            result.persons.append(enfant)

            # Chercher les lieux
            lieux = PATTERNS["lieu"].findall(text)
            if lieux:
                enfant.lieu_naissance = lieux[0]

            # --- Parents ---
            for pat_key in ("fils_de", "fille_de"):
                mp = PATTERNS[pat_key].search(text)
                if mp:
                    pere_name, mere_name = mp.group(1), mp.group(2)
                    pere_nom, pere_prenom = _split_nom_prenom(pere_name)
                    mere_nom, mere_prenom = _split_nom_prenom(mere_name)

                    pere = Person(nom=pere_nom, prenom=pere_prenom, source_document=source)
                    mere = Person(nom=mere_nom, prenom=mere_prenom, source_document=source)
                    result.persons.extend([pere, mere])

                    result.relations.append(FamilyRelation(parent_id=pere.id, enfant_id=enfant.id))
                    result.relations.append(FamilyRelation(parent_id=mere.id, enfant_id=enfant.id))
                    break

        result.confidence = 0.85 if result.persons else 0.1
        return result


# ───────────────────────────── LLM Extractor ──────────────────────────────

LLM_PROMPT = """Tu es un expert en généalogie française. Analyse cet acte d'état civil et extrais les informations en JSON STRICT (sans markdown, sans commentaire).

Format attendu :
{{
  "document_type": "naissance|mariage|deces",
  "personnes": [
    {{
      "role": "enfant|pere|mere|conjoint1|conjoint2|defunt",
      "prenom": "",
      "nom": "",
      "date_naissance": "JJ mois AAAA ou null",
      "lieu_naissance": "ville ou null",
      "date_deces": "JJ mois AAAA ou null",
      "profession": "ou null"
    }}
  ]
}}

Acte :
{texte}"""


class LLMExtractor:
    """
    Extraction via LLM local (Ollama) ou API Anthropic.
    Préféré pour les actes complexes ou manuscrits.
    """

    def __init__(self, model: str = "mistral", backend: str = "ollama"):
        self.model = model
        self.backend = backend

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        prompt = LLM_PROMPT.format(texte=text[:3000])  # limite de contexte
        raw_json = self._call_llm(prompt)
        return self._parse_llm_response(raw_json, text, source)

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        elif self.backend == "anthropic":
            return self._call_anthropic(prompt)
        raise ValueError(f"Backend LLM inconnu : {self.backend}")

    def _call_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _parse_llm_response(self, raw: str, original_text: str, source: str) -> ExtractionResult:
        result = ExtractionResult(document_path=source, raw_text=original_text)
        try:
            # Nettoyer les balises markdown si présentes
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            data = json.loads(clean)
            result.document_type = data.get("document_type")

            role_map: dict[str, Person] = {}
            for p_data in data.get("personnes", []):
                person = Person(
                    nom=p_data.get("nom", "").upper(),
                    prenom=p_data.get("prenom", "").title(),
                    date_naissance=p_data.get("date_naissance"),
                    lieu_naissance=p_data.get("lieu_naissance"),
                    date_deces=p_data.get("date_deces"),
                    profession=p_data.get("profession"),
                    source_document=source,
                )
                result.persons.append(person)
                role_map[p_data.get("role", "")] = person

            # Construire les relations
            enfant = role_map.get("enfant")
            if enfant:
                for role in ("pere", "mere"):
                    parent = role_map.get(role)
                    if parent:
                        result.relations.append(
                            FamilyRelation(parent_id=parent.id, enfant_id=enfant.id)
                        )
            result.confidence = 0.92
        except (json.JSONDecodeError, KeyError) as e:
            result.errors.append(f"Erreur parsing LLM : {e}")
            result.confidence = 0.1
        return result


# ───────────────────────── Extracteur hybride ─────────────────────────────

class HybridExtractor:
    """
    Essaie d'abord le Regex ; si la confiance est trop faible, bascule sur le LLM.
    """

    def __init__(self, llm_threshold: float = 0.5, llm_model: str = "mistral"):
        self.regex = RegexExtractor()
        self.llm = LLMExtractor(model=llm_model)
        self.threshold = llm_threshold

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        result = self.regex.extract(text, source)
        if result.confidence < self.threshold:
            result = self.llm.extract(text, source)
        return result
