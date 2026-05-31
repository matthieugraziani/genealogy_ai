"""
Persistance dans Neo4j (production).
Nécessite Neo4j >= 5.x et le driver python : pip install neo4j
"""

from __future__ import annotations
import os
from contextlib import contextmanager

from neo4j import GraphDatabase, Session

from ..nlp.entities import Person, FamilyRelation
from .family_graph import FamilyGraph


class Neo4jManager:
    """
    Synchronise le FamilyGraph local vers Neo4j.
    Config via variables d'environnement :
      NEO4J_URI      (défaut : bolt://localhost:7687)
      NEO4J_USER     (défaut : neo4j)
      NEO4J_PASSWORD (obligatoire)
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self._driver.close()

    @contextmanager
    def _session(self):
        with self._driver.session() as session:
            yield session

    # ─────────────────── Schéma ───────────────────

    def create_constraints(self):
        with self._session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
            s.run("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.nom, p.prenom)")

    # ─────────────────── Écriture ───────────────────

    def upsert_person(self, person: Person, session: Session | None = None):
        cypher = """
        MERGE (p:Person {id: $id})
        SET p += $props
        """
        props = person.to_dict()
        props.pop("id", None)
        (session or self._driver.session()).run(cypher, id=person.id, props=props)

    def upsert_relation(self, relation: FamilyRelation, session: Session | None = None):
        cypher = f"""
        MATCH (parent:Person {{id: $parent_id}})
        MATCH (enfant:Person {{id: $enfant_id}})
        MERGE (parent)-[r:{relation.relation_type}]->(enfant)
        """
        (session or self._driver.session()).run(
            cypher,
            parent_id=relation.parent_id,
            enfant_id=relation.enfant_id,
        )

    def sync_graph(self, graph: FamilyGraph):
        """Pousse tout le graphe NetworkX vers Neo4j en une transaction."""
        with self._session() as s:
            with s.begin_transaction() as tx:
                for pid, attrs in graph.G.nodes(data=True):
                    person = Person(id=pid, **{k: v for k, v in attrs.items() if k != "id"})
                    tx.run(
                        "MERGE (p:Person {id: $id}) SET p += $props",
                        id=pid,
                        props={k: v for k, v in attrs.items() if k != "id"},
                    )
                for src, dst, data in graph.G.edges(data=True):
                    rel_type = data.get("type", "PARENT_DE")
                    tx.run(
                        f"MATCH (a:Person {{id: $src}}) MATCH (b:Person {{id: $dst}}) "
                        f"MERGE (a)-[:{rel_type}]->(b)",
                        src=src, dst=dst,
                    )
                tx.commit()

    # ─────────────────── Lecture ───────────────────

    def get_ancestors(self, person_id: str, max_depth: int = 10) -> list[dict]:
        cypher = """
        MATCH path = (anc:Person)-[:PARENT_DE*1..{depth}]->(p:Person {{id: $id}})
        RETURN nodes(path) AS nodes, relationships(path) AS rels
        """.format(depth=max_depth)
        with self._session() as s:
            result = s.run(cypher, id=person_id)
            return [record.data() for record in result]

    def search_person(self, query: str) -> list[dict]:
        cypher = """
        MATCH (p:Person)
        WHERE toLower(p.nom) CONTAINS toLower($q)
           OR toLower(p.prenom) CONTAINS toLower($q)
        RETURN p LIMIT 20
        """
        with self._session() as s:
            result = s.run(cypher, q=query)
            return [record["p"] for record in result]
