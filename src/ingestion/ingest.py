from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from src.ingestion.pdf_loader import read_pdf_pages
from src.ingestion.chunking import chunk_text


#PATIENT_ID_REGEX = re.compile(r"\bP\d{3}\b", re.IGNORECASE)
PATIENT_ID_REGEX = re.compile(r"(P\d{3})", re.IGNORECASE)



def infer_patient_id(filename: str) -> str:
    stem = Path(filename).stem
    m = PATIENT_ID_REGEX.search(stem)
    if m:
        return m.group(1).upper()
    return stem

'''
def infer_patient_id(filename: str) -> str:
    """
    Try to infer patient_id from filename.
    Expected patterns like: P012_report.pdf or report_P012.pdf
    If not found, return the file stem.
    """
    m = PATIENT_ID_REGEX.search(filename)
    if m:
        return m.group(0).upper()
    return Path(filename).stem
'''

def make_chunk_record(
    *,
    doc_id: str,
    patient_id: str,
    page: int,
    local_chunk_id: int,
    global_chunk_id: int,
    text: str,
) -> Dict[str, Any]:
    return {
        "chunk_id": global_chunk_id,
        "local_chunk_id": local_chunk_id,
        "doc_id": doc_id,
        "patient_id": patient_id,
        "page": page,
        "text": text,
    }


def ingest_pdfs(
    input_dir: str,
    output_path: str,
    chunk_size: int,
    overlap: int,
) -> int:
    input_path = Path(input_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in input_path.glob("*.pdf")])
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {input_path.resolve()}")

    global_chunk_id = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for pdf in tqdm(pdf_files, desc="Ingesting PDFs"):
            doc_id = pdf.stem
            patient_id = infer_patient_id(pdf.name)

            pages = read_pdf_pages(str(pdf))
            for page_obj in pages:
                # chunk per page (keeps page metadata correct)
                chunks = chunk_text(page_obj.text, chunk_size=chunk_size, overlap=overlap)

                for c in chunks:
                    rec = make_chunk_record(
                        doc_id=doc_id,
                        patient_id=patient_id,
                        page=page_obj.page_number,
                        local_chunk_id=c.chunk_id,
                        global_chunk_id=global_chunk_id,
                        text=c.text,
                    )
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    global_chunk_id += 1
                    written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Ingest medical report PDFs into chunked JSONL.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Folder containing PDF files")
    parser.add_argument("--output", type=str, default="data/processed/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--chunk_size", type=int, default=900, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap in characters")
    args = parser.parse_args()

    n = ingest_pdfs(
        input_dir=args.input_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"✅ Done. Wrote {n} chunks to {args.output}")


if __name__ == "__main__":
    main()
