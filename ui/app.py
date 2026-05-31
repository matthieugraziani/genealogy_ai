"""
Interface Streamlit — Genealogy AI
Lancement : streamlit run ui/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from GitHub.genealogy_ai.ocr.preprocess import preprocess, save_preprocessed
from GitHub.genealogy_ai.ocr.reader import OCRReader
from GitHub.genealogy_ai.nlp.extractor import HybridExtractor
from GitHub.genealogy_ai.graph.family_graph import FamilyGraph

st.set_page_config(page_title="Genealogy AI", page_icon="🌳", layout="wide")
st.title("🌳 Genealogy AI — Reconnaissance d'actes d'état civil")

# ─── Session state ───
if "graph" not in st.session_state:
    st.session_state.graph = FamilyGraph()
if "ocr" not in st.session_state:
    st.session_state.ocr = OCRReader(backend="paddle")
if "extractor" not in st.session_state:
    st.session_state.extractor = HybridExtractor()

graph: FamilyGraph = st.session_state.graph


# ─── Sidebar ───
with st.sidebar:
    st.header("📁 Importer un document")
    uploaded = st.file_uploader(
        "Acte de naissance, mariage ou décès",
        type=["jpg", "jpeg", "png", "tiff", "pdf"],
    )
    ocr_backend = st.selectbox("Moteur OCR", ["paddle", "easyocr", "tesseract"])
    llm_model = st.selectbox("Modèle LLM", ["mistral", "llama3", "gemma3"])

    if uploaded and st.button("🔍 Analyser le document", type="primary"):
        with st.spinner("Traitement en cours…"):
            # Sauvegarder temporairement
            tmp_path = Path("data/images") / uploaded.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.getvalue())

            # Pipeline
            try:
                processed = save_preprocessed(tmp_path)
                reader = OCRReader(backend=ocr_backend)
                raw_text = reader.read(processed)
                extractor = HybridExtractor(llm_model=llm_model)
                result = extractor.extract(raw_text, source=uploaded.name)
                graph.ingest(result)
                st.session_state["last_result"] = result
                st.success(f"✅ {len(result.persons)} personne(s) extraite(s)")
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()
    st.metric("Personnes", graph.stats()["personnes"])
    st.metric("Relations", graph.stats()["relations"])

    if st.button("💾 Exporter GEDCOM"):
        gedcom = graph.to_gedcom()
        st.download_button("Télécharger .ged", gedcom, "arbre.ged", "text/plain")


# ─── Onglets principaux ───
tab1, tab2, tab3 = st.tabs(["📄 Dernière extraction", "👥 Personnes", "🌳 Arbre"])

with tab1:
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.subheader(f"Document : {r.document_path} — Type : {r.document_type}")
        st.caption(f"Confiance : {r.confidence:.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Texte OCR brut**")
            st.text_area("", r.raw_text, height=200, label_visibility="collapsed")
        with col2:
            st.write("**Personnes extraites**")
            for p in r.persons:
                with st.expander(p.nom_complet or "(sans nom)"):
                    st.json(p.to_dict())
    else:
        st.info("Importez un document pour commencer.")

with tab2:
    search_query = st.text_input("🔎 Recherche par nom")
    persons = graph.find_person(search_query) if search_query else list(graph._persons.values())

    if persons:
        import pandas as pd
        rows = []
        for p in persons[:200]:
            rows.append({
                "Prénom": p.prenom,
                "Nom": p.nom,
                "Naissance": p.date_naissance or "",
                "Lieu naissance": p.lieu_naissance or "",
                "Décès": p.date_deces or "",
                "Source": p.source_document or "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Aucune personne dans le graphe.")

with tab3:
    if graph.G.number_of_nodes() == 0:
        st.info("Importez des documents pour construire l'arbre.")
    else:
        st.write(f"**{graph.G.number_of_nodes()} personnes · {graph.G.number_of_edges()} relations**")

        # Affichage avec pyvis si disponible, sinon matplotlib
        try:
            from pyvis.network import Network

            net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff")
            for node_id, attrs in graph.G.nodes(data=True):
                label = f"{attrs.get('prenom', '')} {attrs.get('nom', '')}\n{attrs.get('date_naissance', '')}"
                net.add_node(node_id, label=label.strip(), color="#9FE1CB", font={"size": 12})
            for src, dst in graph.G.edges():
                net.add_edge(src, dst, color="#0F6E56", arrows="to")
            html = net.generate_html()
            st.components.v1.html(html, height=620)
        except ImportError:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(graph.G, k=2, seed=42)
            labels = {
                n: graph.G.nodes[n].get("prenom", "") + " " + graph.G.nodes[n].get("nom", "")
                for n in graph.G.nodes
            }
            nx.draw(graph.G, pos, labels=labels, ax=ax,
                    node_color="#9FE1CB", edge_color="#0F6E56",
                    node_size=1500, font_size=7, arrows=True)
            st.pyplot(fig)
