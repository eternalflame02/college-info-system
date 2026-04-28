import re
import os

filepath = r"c:\Users\Julia\OneDrive\Desktop\jukia\mini\college-info-system\Docs\Project Report\main.tex"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# Remove hogan2021
pattern_hogan = re.compile(r"\s*\\bibitem\{hogan2021\}.*?pp\.~1--37, 2021\.", re.DOTALL)
text = pattern_hogan.sub("", text)

# Find all table blocks
table_pattern = re.compile(r"(\\begin\{table\}.*?)(\\caption\{.*?\}.*?\\label\{.*?\}.*?\n)(.*?)(\\end\{tabularx?\})(.*?)(\\end\{table\})", re.DOTALL)

def rewrite_table(m):
    begin_tbl = m.group(1)
    caption_part = m.group(2)
    middle = m.group(3)
    end_tabular = m.group(4)
    after_tabular = m.group(5)
    end_tbl = m.group(6)
    
    # remove \vspace*{5pt} from caption_part as it looks bad when below table
    caption_part = re.sub(r"\\vspace\*\{.*?\}\n\s*", "", caption_part)
    
    # return re-ordered
    return begin_tbl + middle + end_tabular + "\n  " + caption_part.strip() + "\n" + after_tabular + end_tbl

text = table_pattern.sub(rewrite_table, text)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated tables and references!")
