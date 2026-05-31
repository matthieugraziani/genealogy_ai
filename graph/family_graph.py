"""
Construction et visualisation du graphe généalogique.
Utilise NetworkX en développement ; Neo4j en production (voir neo4j_manager.py).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json

import networkx as nx
from rapidfuzz import fuzz

from ..nlp.entities import Person, FamilyRelation, ExtractionResult


SIMILARITY_THRESHOLD = 85   # score RapidFuzz au-delà duquel deux personnes sont fusionnées


class FamilyGraph:
    """
    Arbre généalogique stocké dans un DiGraph NetworkX.
    Chaque nœud = Person.id, attributs = champs de Person.
    Chaque arc = FamilyRelation.
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._persons: dict[str, Person] = {}   # id → Person

    # ─────────────────── Ajout d'entités ───────────────────

    def add_person(self, person: Person) -> str:
        """
        Ajoute une personne. Si un doublon probable est détecté (RapidFuzz),
        retourne l'id existant au lieu de créer un nouveau nœud.
        """
        existing_id = self._find_duplicate(person)
        if existing_id:
            self._merge(existing_id, person)
            return existing_id

        self.G.add_node(person.id, **person.to_dict())
        self._persons[person.id] = person
        return person.id

    def add_relation(self, relation: FamilyRelation):
        """Ajoute un arc parent → enfant."""
        if relation.parent_id in self.G and relation.enfant_id in self.G:
            self.G.add_edge(
                relation.parent_id,
                relation.enfant_id,
                type=relation.relation_type,
            )

    def ingest(self, result: ExtractionResult):
        """Intègre le résultat d'une extraction complète dans le graphe."""
        id_map: dict[str, str] = {}   # id original → id dans le graphe
        for person in result.persons:
            new_id = self.add_person(person)
            id_map[person.id] = new_id

        for rel in result.relations:
            new_rel = FamilyRelation(
                parent_id=id_map.get(rel.parent_id, rel.parent_id),
                enfant_id=id_map.get(rel.enfant_id, rel.enfant_id),
                relation_type=rel.relation_type,
            )
            self.add_relation(new_rel)

    # ─────────────────── Déduplication ───────────────────

    def _find_duplicate(self, person: Person) -> Optional[str]:
        """Recherche un doublon par similarité de noms (RapidFuzz)."""
        candidate = person.nom_complet.lower()
        for pid, existing in self._persons.items():
            score = fuzz.token_sort_ratio(candidate, existing.nom_complet.lower())
            if score >= SIMILARITY_THRESHOLD:
                # Vérification supplémentaire sur la date de naissance si disponible
                if person.date_naissance and existing.date_naissance:
                    if person.date_naissance != existing.date_naissance:
                        continue
                return pid
        return None

    def _merge(self, existing_id: str, new_person: Person):
        """Enrichit un nœud existant avec les informations manquantes."""
        existing = self._persons[existing_id]
        for attr in ("date_naissance", "lieu_naissance", "date_deces", "profession"):
            if getattr(existing, attr) is None and getattr(new_person, attr) is not None:
                setattr(existing, attr, getattr(new_person, attr))
                self.G.nodes[existing_id][attr] = getattr(new_person, attr)

    # ─────────────────── Requêtes ───────────────────

    def get_ancestors(self, person_id: str, depth: int = 10) -> nx.DiGraph:
        """Retourne le sous-graphe des ancêtres jusqu'à `depth` générations."""
        nodes = nx.ancestors(self.G, person_id) | {person_id}
        return self.G.subgraph(nodes).copy()

    def get_descendants(self, person_id: str) -> nx.DiGraph:
        """Retourne le sous-graphe des descendants."""
        nodes = nx.descendants(self.G, person_id) | {person_id}
        return self.G.subgraph(nodes).copy()

    def find_person(self, query: str) -> list[Person]:
        """Recherche floue par nom."""
        results = []
        for person in self._persons.values():
            if fuzz.partial_ratio(query.lower(), person.nom_complet.lower()) >= 70:
                results.append(person)
        return sorted(results, key=lambda p: fuzz.ratio(query.lower(), p.nom_complet.lower()), reverse=True)

    def stats(self) -> dict:
        return {
            "personnes": self.G.number_of_nodes(),
            "relations": self.G.number_of_edges(),
            "composantes": nx.number_weakly_connected_components(self.G),
        }

    # ─────────────────── Sérialisation ───────────────────

    def to_json(self) -> str:
        data = nx.node_link_data(self.G)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "FamilyGraph":
        g = cls()
        data = json.loads(json_str)
        g.G = nx.node_link_graph(data, directed=True)
        for node_id, attrs in g.G.nodes(data=True):
            g._persons[node_id] = Person(id=node_id, **{k: v for k, v in attrs.items() if k != "id"})
        return g

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FamilyGraph":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ─────────────────── Visualisation (matplotlib) ───────────────────

    def draw(self, max_nodes: int = 50, output_path: Optional[str] = None):
        import matplotlib.pyplot as plt

        sub = self.G
        if self.G.number_of_nodes() > max_nodes:
            nodes = list(self.G.nodes)[:max_nodes]
            sub = self.G.subgraph(nodes)

        labels = {
            n: self.G.nodes[n].get("prenom", "") + "\n" + self.G.nodes[n].get("nom", "")
            for n in sub.nodes
        }
        pos = nx.spring_layout(sub, k=2, seed=42)
        plt.figure(figsize=(14, 10))
        nx.draw(
            sub, pos, labels=labels,
            node_size=1800, node_color="#9FE1CB",
            font_size=8, font_weight="bold",
            edge_color="#0F6E56", arrows=True, arrowsize=15,
        )
        plt.title("Arbre généalogique", fontsize=14)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        else:
            plt.show()

    # ─────────────────── Export GEDCOM ───────────────────

    def to_gedcom(self) -> str:
        """Export minimal au format GEDCOM 5.5.1."""
        lines = ["0 HEAD", "1 GEDC", "2 VERS 5.5.1", "1 CHAR UTF-8"]
        id_to_ged: dict[str, str] = {}
        for i, (pid, person) in enumerate(self._persons.items(), start=1):
            ged_id = f"I{i:04d}"
            id_to_ged[pid] = ged_id
            lines.append(f"0 @{ged_id}@ INDI")
            lines.append(f"1 NAME {person.prenom} /{person.nom}/")
            if person.date_naissance:
                lines.append("1 BIRT")
                lines.append(f"2 DATE {person.date_naissance.upper()}")
            if person.lieu_naissance:
                lines.append(f"2 PLAC {person.lieu_naissance}")
            if person.date_deces:
                lines.append("1 DEAT")
                lines.append(f"2 DATE {person.date_deces.upper()}")

        lines.append("0 TRLR")
        return "\n".join(lines)
