import pdfplumber
import re

pdf_path = r'd:\My\University\NCKH\Code\ViVQAv2\docs\openVqa.pdf'
output_path = r'd:\My\University\NCKH\Code\ViVQAv2\docs\extracted_sections.txt'

with pdfplumber.open(pdf_path) as pdf:
    all_text = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text.append(f"--- PAGE {i+1} ---\n{text}")
    
    full_text = "\n".join(all_text)

# Find section 6 content
# Look for patterns like "6.1" or "6 " or "Experiment"
lines = full_text.split("\n")
in_section = False
section_lines = []
for i, line in enumerate(lines):
    # Start capturing at section 6 or "Experiment"
    if re.search(r'\b6\.?\s*(Experiment|Result|Evaluation)', line, re.IGNORECASE) or \
       re.search(r'^6\s+', line) or \
       re.search(r'\b6\.1\b', line):
        in_section = True
    
    # Stop at section 7
    if in_section and re.search(r'^7\s+|^7\.', line):
        section_lines.append(line)
        # capture a few more lines
        for j in range(i+1, min(i+5, len(lines))):
            section_lines.append(lines[j])
        break
    
    if in_section:
        section_lines.append(line)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(section_lines))

print(f"Extracted {len(section_lines)} lines")
print("First 200 lines:")
print("\n".join(section_lines[:200]))
