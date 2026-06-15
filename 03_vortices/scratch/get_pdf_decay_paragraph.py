import pypdf

reader = pypdf.PdfReader("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/Documents/Sorting in grain flows at the lee side of dunes.pdf")

for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if "exponential" in text.lower() and "decay" in text.lower():
        print(f"=== Page {i+1} ===")
        # Print lines around the keyword
        lines = text.split("\n")
        for j, line in enumerate(lines):
            if "exponential" in line.lower() or "decay" in line.lower():
                start_line = max(0, j - 4)
                end_line = min(len(lines), j + 5)
                print(f"--- Line {j+1} ---")
                for k in range(start_line, end_line):
                    print(f"  {k+1}: {lines[k]}")
