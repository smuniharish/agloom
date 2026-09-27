"""Small, inspectable lexical index over bundled enterprise knowledge."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

WORDS = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "with",
}
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "documents.json"


def tokens(text: str) -> set[str]:
    return set(WORDS.findall(text.casefold())) - STOP_WORDS


@dataclass(frozen=True)
class Passage:
    id: str
    document_id: str
    title: str
    heading: str
    owner: str
    updated: str
    text: str

    def citation(self) -> dict[str, str]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "heading": self.heading,
            "owner": self.owner,
            "updated": self.updated,
            "excerpt": self.text,
        }


class KnowledgeBase:
    def __init__(self, path: Path = DATA_FILE) -> None:
        documents = json.loads(path.read_text(encoding="utf-8"))
        self.passages: dict[str, Passage] = {}
        for document in documents:
            for index, section in enumerate(document["sections"], start=1):
                passage = Passage(
                    id=f"{document['id']}-{index}",
                    document_id=document["id"],
                    title=document["title"],
                    heading=section["heading"],
                    owner=document["owner"],
                    updated=document["updated"],
                    text=section["text"],
                )
                if passage.id in self.passages:
                    raise ValueError(f"duplicate passage ID: {passage.id}")
                self.passages[passage.id] = passage
        self._terms = {
            passage.id: tokens(
                f"{passage.title} {passage.title} {passage.heading} {passage.text}"
            )
            for passage in self.passages.values()
        }
        self._frequency = {
            word: sum(word in terms for terms in self._terms.values())
            for terms in self._terms.values()
            for word in terms
        }

    def search(self, query: str, limit: int = 4) -> list[Passage]:
        query_terms = tokens(query)
        if not query_terms:
            return []
        ranked: list[tuple[float, Passage]] = []
        for passage in self.passages.values():
            overlap = query_terms & self._terms[passage.id]
            if not overlap:
                continue
            score = sum(
                math.log(1 + len(self.passages) / self._frequency[word])
                for word in overlap
            )
            score += 2 * len(query_terms & tokens(passage.title))
            score += 1.5 * len(query_terms & tokens(passage.heading))
            ranked.append((score, passage))
        ranked.sort(key=lambda entry: (-entry[0], entry[1].id))
        return [passage for _, passage in ranked[:limit]]

    def get(self, passage_id: str) -> Passage | None:
        return self.passages.get(passage_id)

    def documents(self) -> list[dict[str, str]]:
        unique: dict[str, dict[str, str]] = {}
        for passage in self.passages.values():
            unique[passage.document_id] = {
                "id": passage.document_id,
                "title": passage.title,
                "owner": passage.owner,
                "updated": passage.updated,
            }
        return list(unique.values())
