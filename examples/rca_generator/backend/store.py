"""Immutable, incident-scoped evidence loaded from local JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Kind = Literal["incident", "log", "metric"]


class Record(BaseModel):
    id: str = Field(min_length=1)
    kind: Kind
    timestamp: str
    text: str = Field(min_length=1)


class Incident(BaseModel):
    id: str = Field(min_length=1)
    title: str
    description: str
    service: str
    severity: str
    status: str
    started_at: str
    impact: str
    records: list[Record] = Field(min_length=1)

    @model_validator(mode="after")
    def check_records(self) -> Incident:
        ids = [item.id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate record IDs in {self.id}")
        return self


class IncidentStore:
    def __init__(self, path: Path | None = None) -> None:
        source = path or Path(__file__).parent / "data" / "incidents.json"
        incidents = [
            Incident.model_validate(row)
            for row in json.loads(source.read_text(encoding="utf-8"))
        ]
        self._incidents = {incident.id: incident for incident in incidents}
        if len(self._incidents) != len(incidents):
            raise ValueError("duplicate incident IDs")

    def list(self) -> list[Incident]:
        return list(self._incidents.values())

    def get(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    def records(self, incident_id: str, kind: Kind) -> list[Record]:
        incident = self.get(incident_id)
        return (
            [record for record in incident.records if record.kind == kind]
            if incident
            else []
        )
