import pypdf
import re

reader = pypdf.PdfReader("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""

lines = text.split('\n')
matches = []
for line in lines:
    if re.search(r'counter|backflow|recirc|shear', line, re.IGNORECASE):
        matches.append(line)

print(f"Found {len(matches)} matches. Showing first 30:")
for m in matches[:30]:
    print("-", m.strip())
