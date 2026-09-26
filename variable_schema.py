"""The fixed set of critical regulatory variables extracted from every
uploaded guidance PDF. Each entry drives one retrieve -> extract ->
guardrail pass through the graph (graph.py): `search_query` feeds hybrid
retrieval (retrieval.py), `extraction_instructions` is slotted into the
extraction prompt (extraction.py). Not every document will discuss every
variable — the pipeline reports "not found in document" per-variable
rather than guessing, same anti-hallucination stance as the rest of the app.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableDef:
    key: str
    label: str
    search_query: str
    extraction_instructions: str
    is_numeric: bool = False


CRITICAL_VARIABLES: list[VariableDef] = [
    VariableDef(
        key="regulatory_authority",
        label="Applicable Regulatory Authority / Scope",
        search_query="regulatory authority scope applicability of this guidance",
        extraction_instructions=(
            "Identify the regulatory authority that issued this guidance and the scope of "
            "products/applicants it applies to."
        ),
    ),
    VariableDef(
        key="storage_conditions",
        label="Storage Conditions",
        search_query="recommended storage temperature and humidity conditions",
        extraction_instructions="Extract the required storage temperature/humidity conditions.",
        is_numeric=True,
    ),
    VariableDef(
        key="shelf_life_retest_period",
        label="Shelf-life / Retest Period",
        search_query="shelf life retest period expiry dating months years",
        extraction_instructions="Extract the shelf-life or retest period requirement.",
        is_numeric=True,
    ),
    VariableDef(
        key="change_notification_timeline",
        label="Post-Approval Change Notification Timeline",
        search_query="timeline to notify regulatory authority of a post-approval change days months",
        extraction_instructions=(
            "Extract the timeline within which a post-approval change must be notified/reported "
            "to the regulatory authority."
        ),
        is_numeric=True,
    ),
    VariableDef(
        key="batch_size_scale_up",
        label="Batch Size / Scale-up Limits",
        search_query="batch size scale up limits fold increase manufacturing",
        extraction_instructions="Extract any batch size or scale-up limit requirements.",
        is_numeric=True,
    ),
    VariableDef(
        key="stability_study_duration",
        label="Stability Study Duration Requirements",
        search_query="stability study duration real time accelerated conditions months",
        extraction_instructions="Extract the required duration/conditions for stability studies.",
        is_numeric=True,
    ),
    VariableDef(
        key="submission_review_timeline",
        label="Submission / Review Timeline",
        search_query="review timeline for submission approval days months",
        extraction_instructions="Extract the regulatory review/approval timeline for a submission.",
        is_numeric=True,
    ),
    VariableDef(
        key="dossier_format",
        label="Required Dossier Format (CTD Module)",
        search_query="dossier format CTD module common technical document structure",
        extraction_instructions="Extract the required dossier/submission format or CTD module reference.",
    ),
    VariableDef(
        key="change_classification",
        label="Change Classification Criteria (Major/Minor/Moderate)",
        search_query="classification of change major minor moderate criteria",
        extraction_instructions="Extract the criteria used to classify a change as major/minor/moderate.",
    ),
    VariableDef(
        key="packaging_requirements",
        label="Packaging / Container-Closure Requirements",
        search_query="packaging container closure system requirements",
        extraction_instructions="Extract packaging or container-closure system requirements.",
    ),
    VariableDef(
        key="labeling_requirements",
        label="Labeling Requirements",
        search_query="labeling requirements label content package insert",
        extraction_instructions="Extract labeling/package-insert content requirements.",
    ),
    VariableDef(
        key="testing_requirements",
        label="Analytical / Testing Requirements for a Change",
        search_query="analytical testing requirements to support a change",
        extraction_instructions="Extract the analytical/testing studies required to support a change.",
    ),
    VariableDef(
        key="applicable_fees",
        label="Applicable Fees",
        search_query="fee payable for submission application change",
        extraction_instructions="Extract any fee amounts payable for a submission or change application.",
        is_numeric=True,
    ),
    VariableDef(
        key="effective_date",
        label="Guidance Effective Date / Applicability",
        search_query="effective date of this guidance applicability date issued",
        extraction_instructions="Extract the effective/issue date and applicability of this guidance.",
    ),
]

VARIABLES_BY_KEY: dict[str, VariableDef] = {v.key: v for v in CRITICAL_VARIABLES}
