# backend/kb_loader.py

from pathlib import Path

KB_PATH = Path("knowledge_base/kb.txt")


def load_kb_sections():
    """
    Reads the KB file and splits it into smaller,
    semantically useful chunks.
    """

    raw_text = KB_PATH.read_text(encoding="utf-8")

    sections = raw_text.split("========================================")

    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue

        # Further split long sections into paragraphs
        paragraphs = section.split("\n\n")
        for para in paragraphs:
            cleaned = para.strip()
            if len(cleaned) > 80:
                chunks.append(cleaned)

    return chunks
