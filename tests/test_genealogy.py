"""
Tests unitaires — Genealogy AI
Lancement : pytest tests/ -v
"""

import pytest
from genealogy_ai.nlp.entities import Person, FamilyRelation, ExtractionResult
from genealogy_ai.nlp.extractor import RegexExtractor
from genealogy_ai.graph.family_graph import FamilyGraph


# ─────────────────── Fixtures ───────────────────

ACTE_NAISSANCE = """
Le 12 mars 1884 est né Jean Dupont,
fils de Pierre Dupont et Marie Martin,
à Paris.
"""

ACTE_NAISSANCE_2 = """
Le 3 juin 1912 est née Louise Dupont,
fille de Jean Dupont et Marguerite Leblanc,
à Lyon.
"""


@pytest.fixture
def extractor():
    return RegexExtractor()


@pytest.fixture
def graph():
    return FamilyGraph()


# ─────────────────── Tests RegexExtractor ───────────────────

def test_extraction_nom(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    assert any("Dupont" in p.nom for p in result.persons)


def test_extraction_date_naissance(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    enfant = next((p for p in result.persons if p.nom == "DUPONT" and p.prenom == "Jean"), None)
    assert enfant is not None
    assert "1884" in enfant.date_naissance


def test_extraction_parents(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    noms = [p.nom for p in result.persons]
    assert "DUPONT" in noms  # père
    assert "MARTIN" in noms  # mère


def test_extraction_relations(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    assert len(result.relations) == 2


def test_type_document_naissance(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    assert result.document_type == "naissance"


def test_confidence_haute(extractor):
    result = extractor.extract(ACTE_NAISSANCE)
    assert result.confidence >= 0.5


def test_texte_vide(extractor):
    result = extractor.extract("")
    assert result.persons == []
    assert result.confidence < 0.5


# ─────────────────── Tests FamilyGraph ───────────────────

def test_add_person(graph):
    p = Person(nom="DUPONT", prenom="Jean")
    pid = graph.add_person(p)
    assert pid in graph.G


def test_deduplication(graph):
    p1 = Person(nom="DUPONT", prenom="Jean")
    p2 = Person(nom="DUPONT", prenom="Jean")   # même personne
    pid1 = graph.add_person(p1)
    pid2 = graph.add_person(p2)
    assert pid1 == pid2
    assert graph.G.number_of_nodes() == 1


def test_no_dedup_different_persons(graph):
    p1 = Person(nom="DUPONT", prenom="Jean")
    p2 = Person(nom="MARTIN", prenom="Marie")
    graph.add_person(p1)
    graph.add_person(p2)
    assert graph.G.number_of_nodes() == 2


def test_ingest_two_generations(graph):
    extractor = RegexExtractor()
    r1 = extractor.extract(ACTE_NAISSANCE)
    r2 = extractor.extract(ACTE_NAISSANCE_2)
    graph.ingest(r1)
    graph.ingest(r2)
    stats = graph.stats()
    # Jean Dupont apparaît dans les deux actes → dédupliqué
    assert stats["personnes"] < (len(r1.persons) + len(r2.persons))


def test_gedcom_export(graph):
    p = Person(nom="DUPONT", prenom="Jean", date_naissance="12 mars 1884")
    graph.add_person(p)
    gedcom = graph.to_gedcom()
    assert "Jean /DUPONT/" in gedcom
    assert "12 MARS 1884" in gedcom
    assert "TRLR" in gedcom


def test_json_roundtrip(graph):
    p = Person(nom="DUPONT", prenom="Jean")
    graph.add_person(p)
    json_str = graph.to_json()
    g2 = FamilyGraph.from_json(json_str)
    assert g2.G.number_of_nodes() == 1


def test_find_person(graph):
    extractor = RegexExtractor()
    result = extractor.extract(ACTE_NAISSANCE)
    graph.ingest(result)
    found = graph.find_person("dupont")
    assert len(found) > 0
