from __future__ import annotations

from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel


class Filing(BaseModel):
    cik: str
    accession_number: str
    form: str
    filed_at: date
    report_date: date | None = None
    primary_document: str
    description: str = ""

    @property
    def filing_url(self) -> str:
        compact_accession = self.accession_number.replace("-", "")
        cik_number = str(int(self.cik))
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_number}/{compact_accession}/{self.primary_document}"
        )


class SecEdgarClient:
    """SEC submissions client with the required descriptive User-Agent."""

    def __init__(
        self,
        user_agent: str,
        *,
        base_url: str = "https://data.sec.gov",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT must identify the application and include a contact email"
            )
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=20,
            transport=transport,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            },
        )

    async def company_submissions(self, cik: str | int) -> dict[str, Any]:
        normalized = str(cik).strip().removeprefix("CIK").zfill(10)
        response = await self.client.get(f"/submissions/CIK{normalized}.json")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def recent_filings(
        self,
        cik: str | int,
        *,
        forms: set[str] | None = None,
        limit: int = 100,
    ) -> list[Filing]:
        payload = await self.company_submissions(cik)
        normalized = str(cik).strip().removeprefix("CIK").zfill(10)
        recent = payload.get("filings", {}).get("recent", {})
        accession_numbers = recent.get("accessionNumber", [])
        filings: list[Filing] = []
        for index, accession in enumerate(accession_numbers):
            form = recent.get("form", [])[index]
            if forms and form not in forms:
                continue
            report_date_raw = recent.get("reportDate", [""])[index]
            filings.append(
                Filing(
                    cik=normalized,
                    accession_number=accession,
                    form=form,
                    filed_at=date.fromisoformat(recent.get("filingDate", [])[index]),
                    report_date=(
                        date.fromisoformat(report_date_raw) if report_date_raw else None
                    ),
                    primary_document=recent.get("primaryDocument", [])[index],
                    description=recent.get("primaryDocDescription", [""])[index],
                )
            )
            if len(filings) >= limit:
                break
        return filings

    async def close(self) -> None:
        await self.client.aclose()
