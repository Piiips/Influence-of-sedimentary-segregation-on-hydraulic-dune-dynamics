import pypdf
import re

reader = pypdf.PdfReader("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf")

full_text = ""
for page_num, page in enumerate(reader.pages):
    ptext = page.extract_text() or ""
    if "7.2." in ptext or "Counterflow effects" in ptext:
        print(f"--- Page {page_num + 1} ---")
        # Let's print paragraphs containing 7.2 or counterflow
        lines = ptext.split("\n")
        start = False
        count = 0
        for line in lines:
            if "7.2" in line or "Counterflow" in line:
                start = True
            if start:
                print(line)
                count += 1
                if count > 45:
                    break
