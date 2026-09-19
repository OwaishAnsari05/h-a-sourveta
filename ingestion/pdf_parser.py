import os
import re
import pymupdf


def clean_text(text):
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    return text.strip()


def remove_headers(text,header_candidates):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line in header_candidates:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def inspect_pages(pages,page_numbers):
    for page_number in page_numbers:
        if page_number < 1 or page_number > len(pages):
            continue

        page = pages[page_number - 1]
        print(f"\n===== PAGE {page_number} =====")
        print(page["text"][:700])


def parse_pdf(pdf_path,document_id,source=None,header_candidates=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if header_candidates is None:
        header_candidates = {
            "Annual Report 2024–2025",
            "Tata Industries Limited",
        }

    if source is None:
        source = os.path.basename(pdf_path)

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number,page in enumerate(doc,start=1):
        raw_text = page.get_text()
        cleaned_text = clean_text(raw_text)
        cleaned_text = remove_headers(
            cleaned_text,
            header_candidates,
        )

        pages.append({
            "document_id": document_id,
            "source": source,
            "page_number": page_number,
            "text": cleaned_text,
        })

    doc.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/documents/Tata_annual_report.pdf"
    document_id = "tata_annual_report_2024_25"

    pages = parse_pdf(
        pdf_path,
        document_id=document_id,
    )

    print(f"total pages: {len(pages)}")

    inspect_pages(
        pages,
        [5,51,101,151],
    )