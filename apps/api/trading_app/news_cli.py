from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .domain import NewsEvent
from .entities import EntityCatalog
from .news_evaluation import build_label_set, evaluate_label_set
from .news_review import (
    adjudicate_label_sets,
    assign_reviewer_packets,
    review_agreement,
)
from .point_in_time_news import PointInTimeNewsIntelligence


def reclassify_news(
    input_path: str,
    output_path: str,
    *,
    catalog_path: str | None = None,
    report_path: str | None = None,
) -> dict[str, object]:
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must differ so the source archive remains immutable")

    catalog = EntityCatalog.load(catalog_path)
    intelligence = PointInTimeNewsIntelligence(
        entity_catalog=catalog,
        drop_exact_duplicates=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    old_primary: Counter[str] = Counter()
    new_primary: Counter[str] = Counter()
    secondary: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    records = 0
    numeric_records = 0
    changed_primary = 0

    with source.open(encoding="utf-8") as input_handle, destination.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            try:
                original = NewsEvent.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid NewsEvent at {source}:{line_number}") from error

            enriched = intelligence.enrich(
                symbol=original.symbol,
                headline=original.headline,
                summary=original.summary,
                source=original.source,
                event_time=original.event_time,
                knowledge_time=original.knowledge_time,
            )
            if enriched is None:
                raise RuntimeError("Reclassification unexpectedly dropped a source record")
            enriched = enriched.model_copy(update={"id": original.id})
            output_handle.write(enriched.model_dump_json() + "\n")

            records += 1
            old_primary[original.event_type] += 1
            new_primary[enriched.event_type] += 1
            secondary.update(enriched.secondary_event_types)
            relations.update(entity.relation.value for entity in enriched.entities)
            sources[enriched.source] += 1
            if original.event_type != enriched.event_type:
                changed_primary += 1
            if any(key != "impact_direction" for key in enriched.event_attributes):
                numeric_records += 1

    if records == 0:
        raise ValueError(f"No news records found in {source}")

    report_destination = Path(report_path) if report_path else Path(f"{destination}.report.json")
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_reclassification_audit",
        "records": records,
        "changed_primary_event_type": changed_primary,
        "numeric_attribute_records": numeric_records,
        "old_primary_event_types": dict(sorted(old_primary.items())),
        "new_primary_event_types": dict(sorted(new_primary.items())),
        "secondary_event_types": dict(sorted(secondary.items())),
        "entity_relations": dict(sorted(relations.items())),
        "sources": dict(sorted(sources.items())),
        "old_other_rate": old_primary.get("other", 0) / records,
        "new_other_rate": new_primary.get("other", 0) / records,
        "catalog": catalog.summary(),
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
        },
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
        },
        "report": str(report_destination),
        "point_in_time_controls": {
            "event_ids_preserved": True,
            "event_time_preserved": True,
            "knowledge_time_preserved": True,
            "catalog_relationships_filtered_by_knowledge_time": True,
            "source_archive_overwritten": False,
        },
        "limitations": [
            "Reclassification can use only headline and summary text already present in the source archive.",
            "Relationships are linked only from the explicit catalog; co-mentions do not create relationships.",
            "Undated catalog relationships are treated as active for the full archive.",
            "A new historical backfill is required to recover ticker copies lost by older cross-symbol deduplication.",
            "Deterministic extraction is an auditable baseline, not a substitute for labelled NLP evaluation.",
        ],
    }
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit, label, and migrate point-in-time news enrichment archives"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    reclassify = subparsers.add_parser(
        "reclassify",
        help="Reclassify an existing NewsEvent JSON Lines archive without changing timestamps",
    )
    reclassify.add_argument("--input", required=True, help="Existing NewsEvent JSON Lines")
    reclassify.add_argument("--output", required=True, help="New enriched NewsEvent JSON Lines")
    reclassify.add_argument("--catalog", help="Optional explicit issuer relationship catalog")
    reclassify.add_argument("--report", help="Optional audit report JSON path")

    catalog = subparsers.add_parser("validate-catalog", help="Validate and summarize a catalog")
    catalog.add_argument("--catalog", required=True)

    sample = subparsers.add_parser(
        "sample-labels",
        help="Create a deterministic stratified CSV for human event/entity labels",
    )
    sample.add_argument("--input", required=True, help="Enriched NewsEvent JSON Lines")
    sample.add_argument("--output", required=True, help="Label-set CSV")
    sample.add_argument("--size", type=int, default=400)
    sample.add_argument("--minimum-per-stratum", type=int, default=3)
    sample.add_argument("--seed", default="news-evaluation-v1")
    sample.add_argument("--metadata", help="Optional label-set metadata JSON path")

    assign = subparsers.add_parser(
        "assign-reviewers",
        help="Create balanced independent reviewer packets from a pending label template",
    )
    assign.add_argument("--labels", required=True, help="Untouched pending label-set CSV")
    assign.add_argument("--output-dir", required=True)
    assign.add_argument(
        "--reviewers",
        required=True,
        help="Comma-separated unique reviewer names",
    )
    assign.add_argument("--reviews-per-item", type=int, default=2)
    assign.add_argument("--seed", default="news-review-assignment-v1")
    assign.add_argument("--report", help="Optional assignment report JSON path")

    agreement = subparsers.add_parser(
        "review-agreement",
        help="Measure inter-annotator agreement across independent reviewer CSVs",
    )
    agreement.add_argument("--labels", nargs="+", required=True, help="Reviewer CSV files")
    agreement.add_argument("--output", required=True, help="Agreement report JSON")
    agreement.add_argument("--require-complete", action="store_true")

    adjudicate = subparsers.add_parser(
        "adjudicate-labels",
        help="Create a unanimous-only consensus CSV and retain disputes for human review",
    )
    adjudicate.add_argument("--labels", nargs="+", required=True, help="Reviewer CSV files")
    adjudicate.add_argument("--output", required=True, help="Consensus/adjudication CSV")
    adjudicate.add_argument("--minimum-reviewers", type=int, default=2)
    adjudicate.add_argument("--report", help="Optional adjudication report JSON")

    evaluate = subparsers.add_parser(
        "evaluate-labels",
        help="Measure event and entity precision/recall from a reviewed label-set CSV",
    )
    evaluate.add_argument("--labels", required=True, help="Reviewed label-set CSV")
    evaluate.add_argument("--output", required=True, help="Evaluation report JSON")
    evaluate.add_argument("--metadata", help="Optional label-set metadata JSON path")
    evaluate.add_argument("--require-complete", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "reclassify":
        result = reclassify_news(
            arguments.input,
            arguments.output,
            catalog_path=arguments.catalog,
            report_path=arguments.report,
        )
    elif arguments.command == "validate-catalog":
        result = EntityCatalog.load(arguments.catalog).summary()
    elif arguments.command == "sample-labels":
        result = build_label_set(
            arguments.input,
            arguments.output,
            sample_size=arguments.size,
            minimum_per_stratum=arguments.minimum_per_stratum,
            seed=arguments.seed,
            metadata_path=arguments.metadata,
        )
    elif arguments.command == "assign-reviewers":
        result = assign_reviewer_packets(
            arguments.labels,
            arguments.output_dir,
            [part.strip() for part in arguments.reviewers.split(",")],
            reviews_per_item=arguments.reviews_per_item,
            seed=arguments.seed,
            report_path=arguments.report,
        )
    elif arguments.command == "review-agreement":
        result = review_agreement(
            arguments.labels,
            arguments.output,
            require_complete=arguments.require_complete,
        )
    elif arguments.command == "adjudicate-labels":
        result = adjudicate_label_sets(
            arguments.labels,
            arguments.output,
            minimum_reviewers=arguments.minimum_reviewers,
            report_path=arguments.report,
        )
    else:
        result = evaluate_label_set(
            arguments.labels,
            arguments.output,
            require_complete=arguments.require_complete,
            metadata_path=arguments.metadata,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
