from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class URL:
    original: str
    domain: str
    is_suspicious: bool = False
    suspicion_reasons: list[str] = field(default_factory=list)


@dataclass
class EmailHeaders:
    subject: str
    from_address: str
    to_addresses: list[str]
    date: datetime
    message_id: str = ""
    spf: str = "unknown"
    dkim: str = "unknown"
    dmarc: str = "unknown"


@dataclass
class EmailData:
    headers: EmailHeaders
    body: str
    urls: list[URL] = field(default_factory=list)
    email_id: str = ""
    label: bool | None = None
