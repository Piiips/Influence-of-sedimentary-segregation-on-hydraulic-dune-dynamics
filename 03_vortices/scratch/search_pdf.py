import os
import re

# Try importing common PDF libraries
pdf_text = ""
try:
    import pypdf
    reader = pypdf.PdfReader("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf")
    for page in reader.pages:
        pdf_text += page.extract_text() or ""
    print("Extracted via pypdf successfully.")
except Exception as e:
    print("pypdf failed:", e)

if not pdf_text:
    try:
        import pdfplumber
        with pdfplumber.open("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf") as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text() or ""
        print("Extracted via pdfplumber successfully.")
    except Exception as e:
        print("pdfplumber failed:", e)

if not pdf_text:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf")
        for page in doc:
            pdf_text += page.get_text()
        print("Extracted via fitz successfully.")
    except Exception as e:
        print("fitz failed:", e)

if pdf_text:
    # Find all occurrences of "shear" or "stress" or "cizalla"
    lines = pdf_text.split('\n')
    matches = []
    for line in lines:
        if re.search(r'\bshear\b|\bstress\b', line, re.IGNORECASE):
            matches.append(line)
    
    print(f"\nFound {len(matches)} matching lines. Showing first 40:")
    for m in matches[:40]:
        print("-", m.strip())
else:
    print("Could not extract PDF text.")
