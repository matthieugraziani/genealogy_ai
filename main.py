"""
Point d'entrée principal — Genealogy AI
Usage :
  python main.py --mode api          → lance le serveur FastAPI
  python main.py --mode ui           → lance l'interface Streamlit
  python main.py --file acte.jpg     → traite un fichier en ligne de commande
"""

import argparse
import sys
from pathlib import Path


def run_api(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


def run_ui():
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"], check=True)


def process_file(file_path: str, backend: str = "paddle"):
    from GitHub.genealogy_ai.ocr.preprocess import save_preprocessed
    from GitHub.genealogy_ai.ocr.reader import OCRReader
    from GitHub.genealogy_ai.nlp.extractor import HybridExtractor
    from GitHub.genealogy_ai.graph.family_graph import FamilyGraph

    path = Path(file_path)
    if not path.exists():
        print(f"Fichier introuvable : {file_path}")
        sys.exit(1)

    print(f"→ Prétraitement de {path.name}…")
    processed = save_preprocessed(path)

    print("→ OCR…")
    reader = OCRReader(backend=backend)
    text = reader.read(processed)
    print(f"  Texte extrait ({len(text)} caractères) :")
    print("  " + text[:300].replace("\n", "\n  "))

    print("\n→ Extraction des entités…")
    extractor = HybridExtractor()
    result = extractor.extract(text, source=file_path)
    print(f"  Type de document : {result.document_type}")
    print(f"  Confiance : {result.confidence:.0%}")
    for p in result.persons:
        print(f"  · {p.nom_complet} (naissance : {p.date_naissance or 'N/A'})")

    print("\n→ Construction du graphe…")
    graph = FamilyGraph()
    graph.ingest(result)
    stats = graph.stats()
    print(f"  {stats['personnes']} personne(s), {stats['relations']} relation(s)")

    out_json = Path("data/extracted") / (path.stem + "_graph.json")
    out_gedcom = Path("data/extracted") / (path.stem + ".ged")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    graph.save(out_json)
    out_gedcom.write_text(graph.to_gedcom(), encoding="utf-8")
    print(f"\n  Graphe sauvegardé : {out_json}")
    print(f"  GEDCOM exporté   : {out_gedcom}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genealogy AI")
    parser.add_argument("--mode", choices=["api", "ui"], help="Lance le serveur API ou l'interface Streamlit")
    parser.add_argument("--file", help="Traite un document en ligne de commande")
    parser.add_argument("--backend", default="paddle", choices=["paddle", "easyocr", "tesseract"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.file:
        process_file(args.file, args.backend)
    elif args.mode == "api":
        run_api(args.host, args.port)
    elif args.mode == "ui":
        run_ui()
    else:
        parser.print_help()
