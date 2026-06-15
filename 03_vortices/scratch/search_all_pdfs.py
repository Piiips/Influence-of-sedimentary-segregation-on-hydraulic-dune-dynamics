import os
import re
import pypdf

doc_dir = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents"
files = [f for f in os.listdir(doc_dir) if f.endswith(".pdf")]

keywords = ["shear stress", "shear rate", "separation", "lee", "reattachment"]

for fname in files:
    path = os.path.join(doc_dir, fname)
    print(f"\nSearching in {fname}...")
    try:
        reader = pypdf.PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            ptext = page.extract_text() or ""
            # Check for combinations
            if "shear" in ptext.lower() and ("lee" in ptext.lower() or "separation" in ptext.lower()):
                lines = ptext.split("\n")
                for line in lines:
                    if "shear" in line.lower() and ("stress" in line.lower() or "rate" in line.lower() or "velocity" in line.lower()):
                        print(f"  [Page {i+1}] {line.strip()}")
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
