import httpx
import pytest

from trading_app.sec import SecEdgarClient


@pytest.mark.asyncio
async def test_sec_client_parses_and_filters_recent_filings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == "Trading Research contact@example.com"
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
                        "filingDate": ["2026-01-03", "2026-01-04"],
                        "reportDate": ["2025-12-31", ""],
                        "form": ["10-K", "4"],
                        "primaryDocument": ["aapl-20251231.htm", "ownership.xml"],
                        "primaryDocDescription": ["Annual report", "Ownership report"],
                    }
                }
            },
        )

    client = SecEdgarClient(
        "Trading Research contact@example.com",
        base_url="https://data.sec.test",
        transport=httpx.MockTransport(handler),
    )
    filings = await client.recent_filings("320193", forms={"10-K", "10-Q", "8-K"})
    await client.close()

    assert len(filings) == 1
    assert filings[0].form == "10-K"
    assert filings[0].filing_url.endswith("/aapl-20251231.htm")


def test_sec_client_requires_contact_user_agent() -> None:
    with pytest.raises(ValueError):
        SecEdgarClient("anonymous-script")
