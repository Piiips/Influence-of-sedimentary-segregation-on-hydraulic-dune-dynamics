import pypdf
import re

pdf_path = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf"
reader = pypdf.PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text() or ""

lines = text.split("\n")
matches = []
for line in lines:
    if re.search(r'exponential|decay|fallout|grainfall|jopling|hunter', line, re.IGNORECASE):
        matches.append(line)

print("Matches in 'Sorting in grain flows...':")
for m in matches[:30]:
    print("-", m.strip())
