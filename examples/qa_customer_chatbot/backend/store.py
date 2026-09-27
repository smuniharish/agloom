"""Bundled demo records and deterministic, inspectable policy lookup."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
}


def terms(text: str) -> set[str]:
    return set(WORDS.findall(text.casefold())) - STOP_WORDS


class Order(BaseModel):
    id: str
    status: str
    item: str
    estimated_delivery: str


class Customer(BaseModel):
    id: str
    name: str
    orders: list[Order]


class Policy(BaseModel):
    id: str
    title: str
    text: str

    def citation(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "excerpt": self.text}


class Store:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        customers = [
            Customer.model_validate(row)
            for row in json.loads(
                (data_dir / "customers.json").read_text(encoding="utf-8")
            )
        ]
        policies = [
            Policy.model_validate(row)
            for row in json.loads(
                (data_dir / "policies.json").read_text(encoding="utf-8")
            )
        ]
        self.customers = {row.id: row for row in customers}
        self.policies = {row.id: row for row in policies}
        if len(self.customers) != len(customers) or len(self.policies) != len(policies):
            raise ValueError("Duplicate customer or policy ID in seed data")

    def order(self, customer_id: str, order_id: str | None = None) -> Order | None:
        customer = self.customers.get(customer_id)
        if customer is None:
            return None
        if order_id is None:
            return customer.orders[0] if customer.orders else None
        return next((order for order in customer.orders if order.id == order_id), None)

    def search(self, query: str, limit: int = 3) -> list[Policy]:
        query_terms = terms(query)
        scored = [
            (
                len(
                    query_terms & terms(f"{policy.title} {policy.title} {policy.text}")
                ),
                policy,
            )
            for policy in self.policies.values()
        ]
        return [
            policy
            for score, policy in sorted(
                scored, key=lambda pair: (-pair[0], pair[1].id)
            )[:limit]
            if score > 0
        ]
