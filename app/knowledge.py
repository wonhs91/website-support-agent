from pathlib import Path


def load_company_knowledge() -> str:
    """Load static company/product information from the knowledge folder.

    For now this is a single small blob that is always injected
    into the QA node prompt. When the knowledge base grows, this
    can be replaced with a real retrieval layer.
    """
    root = Path(__file__).resolve().parent.parent
    docs_path = root / "knowledge" / "company.md"
    if docs_path.is_file():
        return docs_path.read_text(encoding="utf-8")
    return ""


COMPANY_KNOWLEDGE = load_company_knowledge()
