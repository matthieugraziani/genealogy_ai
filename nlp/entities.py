"""
Modèles de données pour les personnes et les relations familiales.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class Person:
    """Représente un individu extrait d'un acte d'état civil."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nom: str = ""
    prenom: str = ""
    date_naissance: Optional[str] = None
    lieu_naissance: Optional[str] = None
    date_deces: Optional[str] = None
    lieu_deces: Optional[str] = None
    date_mariage: Optional[str] = None
    lieu_mariage: Optional[str] = None
    profession: Optional[str] = None
    source_document: Optional[str] = None   # chemin du document d'origine

    @property
    def nom_complet(self) -> str:
        return f"{self.prenom} {self.nom}".strip()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __hash__(self):
        return hash(self.id)


@dataclass
class FamilyRelation:
    """Représente un lien familial entre deux personnes."""

    parent_id: str
    enfant_id: str
    relation_type: str = "PARENT_DE"   # PARENT_DE, CONJOINT_DE, SIBLING_OF

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ExtractionResult:
    """Résultat complet de l'extraction depuis un document."""

    document_path: str
    raw_text: str
    persons: list[Person] = field(default_factory=list)
    relations: list[FamilyRelation] = field(default_factory=list)
    document_type: Optional[str] = None   # "naissance", "mariage", "deces"
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)
