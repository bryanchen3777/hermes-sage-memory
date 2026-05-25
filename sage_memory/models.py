from dataclasses import dataclass, field
from typing import Literal, Optional
import uuid
import time


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    timestamp: float = field(default_factory=time.time)
    event_time: Optional[float] = None
    weight: float = 1.0
    source: Literal["user", "inference", "correction"] = "user"
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    is_anchor: bool = False

    def to_dict(self) -> dict:
        return {
            "subject":    self.subject,
            "predicate":  self.predicate,
            "object":     self.object,
            "timestamp":  self.timestamp,
            "event_time": self.event_time,
            "weight":     self.weight,
            "source":     self.source,
            "fact_id":    self.fact_id,
            "session_id": self.session_id,
            "is_anchor":  self.is_anchor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        d.setdefault("event_time", None)
        d.setdefault("is_anchor", False)
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