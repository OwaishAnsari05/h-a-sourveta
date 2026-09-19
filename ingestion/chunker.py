import json
import os
import re
from ingestion.pdf_parser import parse_pdf

CHUNKS_DIR = "data/chunks"
MAX_CHARS = 1800
OVERLAP_CHARS = 300
OUTPUT_PATH = "data/chunks.json"
IMPORTANT_PAGES = {5,101,151}

TABLE_LABELS = {
    "particulars","standalone","consolidated","fy 2024-25","fy 2023-24",
    "march 31, 2025","march 31, 2024","march 31 2025","march 31 2024",
}

def normalize_line(line):
    return re.sub(r"\s+"," ",line.strip())

def is_table_label(line):
    normalized = normalize_line(line)
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in TABLE_LABELS:
        return True
    if re.fullmatch(r"fy\s+\d{4}[-–]\d{2}",lowered):
        return True
    if re.fullmatch(r"march\s+31,?\s+\d{4}",lowered,flags=re.IGNORECASE):
        return True
    return False

def is_heading(line):
    line = normalize_line(line)
    if not line or is_table_label(line):
        return False
    numbered_match = re.match(r"^(\d+)[\.\)]?\s+(.+)$",line)
    if numbered_match:
        title = numbered_match.group(2).strip()
        if len(title) < 3:
            return False
        if re.fullmatch(r"FY\s+\d{4}[-–]\d{2}",title,flags=re.IGNORECASE):
            return False
        if is_table_label(title) or re.fullmatch(r"[\d,.\-()₹`]+",title):
            return False
        return True
    if re.match(r"^[A-Z]\.\s+[A-Za-z]",line):
        return True
    if line.isupper():
        words = line.split()
        if 1 <= len(words) <= 10 and len(line) >= 4 and not is_table_label(line):
            return True
    return False

def get_multiline_heading(lines,index):
    if index >= len(lines):
        return None,0
    current = normalize_line(lines[index])
    if not re.fullmatch(r"\d+",current) or index + 1 >= len(lines):
        return None,0
    next_line = normalize_line(lines[index + 1])
    if not next_line or is_table_label(next_line):
        return None,0
    if not re.match(r"^[A-Z]",next_line) or len(next_line.split()) > 12:
        return None,0
    return f"{current} {next_line}",2

def extract_sections(text):
    lines = text.splitlines()
    sections = []
    current_section = "General"
    current_content = []
    i = 0
    while i < len(lines):
        line = normalize_line(lines[i])
        if not line:
            i += 1
            continue
        multiline_heading,consumed = get_multiline_heading(lines,i)
        if multiline_heading:
            if current_content:
                sections.append({"section":current_section,"text":"\n".join(current_content)})
            current_section = multiline_heading
            current_content = []
            i += consumed
            continue
        if is_heading(line):
            if current_content:
                sections.append({"section":current_section,"text":"\n".join(current_content)})
            current_section = line
            current_content = []
            i += 1
            continue
        current_content.append(line)
        i += 1
    if current_content:
        sections.append({"section":current_section,"text":"\n".join(current_content)})
    return sections

def split_section(text,max_chars=MAX_CHARS,overlap_chars=OVERLAP_CHARS):
    paragraphs = [normalize_line(p) for p in text.split("\n") if normalize_line(p)]
    if not paragraphs:
        return []
    chunks = []
    current = []
    current_length = 0
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if paragraph_length > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            start = 0
            while start < paragraph_length:
                end = min(start + max_chars,paragraph_length)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = end
            continue
        if not current:
            current = [paragraph]
            current_length = paragraph_length
            continue
        proposed_length = current_length + 1 + paragraph_length
        if proposed_length <= max_chars:
            current.append(paragraph)
            current_length = proposed_length
            continue
        chunks.append("\n".join(current))
        overlap = []
        overlap_length = 0
        for previous in reversed(current):
            previous_length = len(previous)
            if overlap_length + previous_length + 1 <= overlap_chars:
                overlap.insert(0,previous)
                overlap_length += previous_length + 1
            else:
                break
        current = overlap + [paragraph]
        current_length = sum(len(p) + 1 for p in current)
    if current:
        chunks.append("\n".join(current))
    return chunks

def create_section_chunks(pages,max_chars=MAX_CHARS,overlap_chars=OVERLAP_CHARS):
    all_chunks = []
    for page in pages:
        page_number = page["page_number"]
        document_id = page["document_id"]
        source = page["source"]
        sections = extract_sections(page["text"])
        section_counter = 0
        for section in sections:
            section_name = section["section"]
            section_text = section["text"].strip()
            if not section_text:
                continue
            section_counter += 1
            chunks = split_section(section_text,max_chars=max_chars,overlap_chars=overlap_chars)
            for chunk_index,chunk_text in enumerate(chunks,start=1):
                chunk_id = f"{document_id}_p{page_number}_s{section_counter}_c{chunk_index}"
                all_chunks.append({
                    "chunk_id":chunk_id,
                    "document_id":document_id,
                    "source":source,
                    "page_number":page_number,
                    "section":section_name,
                    "section_index":section_counter,
                    "chunk_index":chunk_index,
                    "text":chunk_text,
                })
    return all_chunks

def save_chunks(chunks,output_path=OUTPUT_PATH):
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory,exist_ok=True)
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)

def get_document_chunks_path(document_id,chunks_dir=CHUNKS_DIR):
    os.makedirs(chunks_dir,exist_ok=True)
    return os.path.join(chunks_dir,f"{document_id}.json")

def build_document_chunks(pdf_path,document_id,source=None,output_path=None,max_chars=MAX_CHARS,overlap_chars=OVERLAP_CHARS):
    pages = parse_pdf(pdf_path,document_id=document_id,source=source)
    chunks = create_section_chunks(pages,max_chars=max_chars,overlap_chars=overlap_chars)
    if output_path is None:
        output_path = get_document_chunks_path(document_id)
    save_chunks(chunks,output_path)
    return pages,chunks

def run_heading_tests():
    print("\n" + "=" * 70)
    print("HEADING DETECTION TEST")
    print("=" * 70)
    test_lines = [
        "FY 2024-25","FY 2023-24","1. FINANCIAL RESULTS",
        "6 Provisions and Contingencies","BOARD'S REPORT","CONSOLIDATED",
        "STANDALONE","PARTICULARS","2. OPERATIONS OF THE COMPANY",
    ]
    for test_line in test_lines:
        print(f"{test_line!r:<40} -> is_heading = {is_heading(test_line)}")

def inspect_chunks(chunks):
    print("\n" + "=" * 70)
    print("SAMPLE SECTION-AWARE CHUNKS")
    print("=" * 70)
    for chunk in chunks:
        if chunk["page_number"] not in IMPORTANT_PAGES:
            continue
        print("\n" + "-" * 70)
        print("CHUNK ID:     ",chunk["chunk_id"])
        print("PAGE:         ",chunk["page_number"])
        print("SECTION:      ",chunk["section"])
        print("SECTION INDEX:",chunk["section_index"])
        print("CHUNK INDEX:  ",chunk["chunk_index"])
        print("-" * 70)
        print(chunk["text"][:1800])

if __name__ == "__main__":
    pdf_path = "data/documents/Tata_annual_report.pdf"
    document_id = "tata_annual_report_2024_25"
    pages,chunks = build_document_chunks(
        pdf_path,
        document_id=document_id,
        source="Tata_annual_report.pdf",
    )
    print(f"Total pages processed: {len(pages)}")
    print(f"Total section-aware chunks: {len(chunks)}")
    run_heading_tests()
    inspect_chunks(chunks)
    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)
    print(f"Saved {len(chunks)} section-aware chunks.")