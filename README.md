# Genealogy AI

Reconnaissance automatique de documents d'état civil pour construire un arbre généalogique.

## Stack

| Couche | Technologie |
|---|---|
| Prétraitement | OpenCV, Pillow |
| OCR | PaddleOCR (défaut), EasyOCR, Tesseract |
| Extraction NLP | Regex + LLM local (Mistral via Ollama) |
| Déduplication | RapidFuzz |
| Graphe (dev) | NetworkX |
| Graphe (prod) | Neo4j 5.x |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Export | GEDCOM 5.5.1 |

## Installation

```bash
# Cloner et créer l'environnement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Modèle spaCy français (optionnel)
python -m spacy download fr_core_news_lg

# LLM local via Ollama
ollama pull mistral
```

## Utilisation

```bash
# Traiter un fichier directement
python main.py --file data/images/naissance_1884.jpg

# Lancer l'API REST
python main.py --mode api
# → http://localhost:8000/docs

# Lancer l'interface graphique
python main.py --mode ui
```

## API REST — exemples

```bash
# Upload d'un acte
curl -X POST http://localhost:8000/upload \
  -F "file=@acte_naissance.jpg"

# Recherche par nom
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Dupont"}'

# Export GEDCOM
curl http://localhost:8000/tree/gedcom > arbre.ged
```

## Structure du projet

```
genealogy_ai/
├── data/
│   ├── images/          images sources
│   └── extracted/       résultats OCR et graphes JSON
├── ocr/
│   ├── preprocess.py    pipeline OpenCV
│   └── reader.py        wrapper OCR multi-backend
├── nlp/
│   ├── entities.py      modèles de données
│   └── extractor.py     Regex + LLM hybride
├── graph/
│   ├── family_graph.py  graphe NetworkX + dédup + GEDCOM
│   └── neo4j_manager.py persistance Neo4j
├── api/
│   └── main.py          FastAPI
├── ui/
│   └── app.py           Streamlit
├── tests/
│   └── test_genealogy.py
├── main.py
├── pyproject.toml
└── .env.example
```

## Tests

```bash
pytest tests/ -v
```

## Variables d'environnement

Copier `.env.example` → `.env` et renseigner :
- `NEO4J_PASSWORD` pour la persistance Neo4j
- `ANTHROPIC_API_KEY` pour utiliser Claude comme LLM d'extraction
- `OLLAMA_HOST` si Ollama tourne sur un autre hôte
>>>>>>> 5473e0f (initial commit: structure du projet)
