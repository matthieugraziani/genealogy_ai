"""
API REST — Genealogy AI
Routes :
  POST /upload       → traite un document, retourne les entités extraites
  GET  /persons      → liste toutes les personnes du graphe
  GET  /persons/{id}/ancestors  → ancêtres d'une personne
  GET  /tree         → graphe complet en JSON (node-link)
  GET  /tree/gedcom  → export GEDCOM
  POST /search       → recherche floue par nom
"""

from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from ..ocr.preprocess import preprocess, save_preprocessed
from ..ocr.reader import OCRReader
from ..nlp.extractor import HybridExtractor
from ..graph.family_graph import FamilyGraph

app = FastAPI(
    title="Genealogy AI",
    description="Reconnaissance automatique de documents d'état civil → arbre généalogique",
    version="1.0.0",
)

# ─── État global (remplacer par une vraie persistance en prod) ───
graph = FamilyGraph()
ocr = OCRReader(backend="paddle")
extractor = HybridExtractor()


# ─────────────────── Schémas ───────────────────

class SearchRequest(BaseModel):
    query: str


# ─────────────────── Routes ───────────────────

@app.post("/upload", summary="Traiter un document d'état civil")
async def upload_document(file: Annotated[UploadFile, File(description="PDF, JPG, PNG ou TIFF")]):
    if not file.filename:
        raise HTTPException(400, "Nom de fichier manquant")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pdf"}
    if suffix not in allowed:
        raise HTTPException(400, f"Format non supporté : {suffix}. Acceptés : {allowed}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        # 1. Prétraitement
        processed_path = save_preprocessed(tmp_path)

        # 2. OCR
        raw_text = ocr.read(processed_path)
        if not raw_text.strip():
            raise HTTPException(422, "Aucun texte extrait — vérifiez la qualité de l'image")

        # 3. Extraction NLP
        result = extractor.extract(raw_text, source=file.filename)

        # 4. Intégration dans le graphe
        graph.ingest(result)

        return {
            "document": file.filename,
            "type": result.document_type,
            "personnes": [p.to_dict() for p in result.persons],
            "relations": [r.to_dict() for r in result.relations],
            "confiance": result.confidence,
            "stats_graphe": graph.stats(),
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/persons", summary="Lister toutes les personnes")
def list_persons(limit: int = Query(100, le=1000)):
    persons = list(graph._persons.values())[:limit]
    return [p.to_dict() for p in persons]


@app.get("/persons/{person_id}/ancestors", summary="Ancêtres d'une personne")
def get_ancestors(person_id: str, depth: int = Query(10, le=20)):
    if person_id not in graph.G:
        raise HTTPException(404, f"Personne {person_id} introuvable")
    sub = graph.get_ancestors(person_id, depth)
    import networkx as nx
    return nx.node_link_data(sub)


@app.get("/tree", summary="Graphe complet (node-link JSON)")
def get_tree():
    return JSONResponse(content={"graph": graph.to_json()})


@app.get("/tree/gedcom", response_class=PlainTextResponse, summary="Export GEDCOM")
def export_gedcom():
    return graph.to_gedcom()


@app.post("/search", summary="Recherche floue par nom")
def search_person(body: SearchRequest):
    results = graph.find_person(body.query)
    return [p.to_dict() for p in results[:20]]


@app.get("/stats", summary="Statistiques du graphe")
def get_stats():
    return graph.stats()


@app.get("/health")
def health():
    return {"status": "ok"}
