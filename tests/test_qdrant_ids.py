"""Tests de l'indexation idempotente (identifiants de points uuid5)."""
import uuid

import pytest
from qdrant_client import QdrantClient

from maintidoc.indexing.qdrant_store import QdrantStore, make_point_id
from maintidoc.schemas import TableRecord, TextRecord

SHA_V1 = "a" * 64
SHA_V2 = "b" * 64


def test_point_id_is_deterministic_and_valid_uuid():
    pid = make_point_id("SAN-MAN-001", SHA_V1, "p03_text_01")
    assert pid == make_point_id("SAN-MAN-001", SHA_V1, "p03_text_01")
    assert uuid.UUID(pid).version == 5


@pytest.mark.parametrize(
    "other",
    [
        ("SAN-MAN-002", SHA_V1, "p03_text_01"),  # autre document
        ("SAN-MAN-001", SHA_V2, "p03_text_01"),  # nouvelle version du PDF
        ("SAN-MAN-001", SHA_V1, "p03_table_01"),  # autre élément
    ],
)
def test_point_id_changes_when_any_component_changes(other):
    assert make_point_id("SAN-MAN-001", SHA_V1, "p03_text_01") != make_point_id(*other)


class FakeEmbeddings:
    """Remplace BGE-M3 : vecteur constant de la bonne dimension, sans téléchargement."""

    def encode_text(self, text):
        return [0.1] * QdrantStore.TEXT_DIM


@pytest.fixture
def store():
    # Contourne __init__ (qui tente une connexion réseau) : Qdrant en mémoire
    s = QdrantStore.__new__(QdrantStore)
    s.client = QdrantClient(":memory:")
    s.init_collections()
    return s


def _records(sha):
    texts = [TextRecord(document_sha256=sha, page=1, element_id="p01_text_01", modality="text", text="Voyant orange")]
    tables = [
        TableRecord(
            document_sha256=sha, page=2, element_id="p02_table_01", modality="table",
            bbox=[0, 0, 1, 1], rows=[["LED", "Etat"]], text="| LED | Etat |",
        )
    ]
    return texts, tables


def test_reindexing_same_document_does_not_duplicate(store):
    texts, tables = _records(SHA_V1)

    store.index_texts_and_tables(texts, tables, FakeEmbeddings())
    store.index_texts_and_tables(texts, tables, FakeEmbeddings())

    assert store.client.count(QdrantStore.TEXT_COLLECTION).count == 2


def test_search_text_returns_hits_with_payload(store):
    """Régression : ``client.search`` n'existe plus dans qdrant-client 1.20."""
    store.index_texts_and_tables(*_records(SHA_V1), FakeEmbeddings())

    hits = store.search_text("voyant orange", FakeEmbeddings(), top_k=2)

    assert [h.rank for h in hits] == [1, 2]
    assert {h.element_id for h in hits} == {"p01_text_01", "p02_table_01"}
    assert all(h.metadata["content"] for h in hits)


def test_new_document_version_is_kept_separately(store):
    store.index_texts_and_tables(*_records(SHA_V1), FakeEmbeddings())
    store.index_texts_and_tables(*_records(SHA_V2), FakeEmbeddings())

    assert store.client.count(QdrantStore.TEXT_COLLECTION).count == 4
