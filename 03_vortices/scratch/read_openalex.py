import json

with open("/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/scratch/openalex_results.json", "r") as f:
    data = json.load(f)

print(f"Total results found: {len(data.get('results', []))}")
for idx, paper in enumerate(data.get('results', [])):
    title = paper.get('display_name')
    year = paper.get('publication_year')
    doi = paper.get('doi')
    citations = paper.get('cited_by_count')
    print(f"{idx+1}. [{year}] {title}")
    print(f"   DOI: {doi}")
    print(f"   Citations: {citations}\n")
