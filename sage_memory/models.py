from dataclasses import dataclass, field
from typing import Literal
import uuid
import time


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0
    source: Literal["user", "inference", "correction"] = "user"
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "timestamp": self.timestamp,
            "weight": self.weight,
            "source": self.source,
            "fact_id": self.fact_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**d)


@dataclass
class ContextResult:
    facts: list[Fact]
    chains: list[list[Fact]]
    summary: str
    token_estimate: int

    @property
    def is_empty(self) -> bool:
        return len(self.facts) == 0