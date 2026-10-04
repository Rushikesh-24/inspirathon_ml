#!/usr/bin/env python
"""Convert submission_notebook.py to a Kaggle-compatible .ipynb notebook."""
import json, re

with open("submission_notebook.py", "r") as f:
    content = f.read()

# Split on # %% markers
cells = []
# Remove the shebang line
content = re.sub(r'^#!/usr/bin/env python\n', '', content)

# Split by cell markers
parts = re.split(r'\n# %%(?:\s*\[markdown\])?\n', content)

for i, part in enumerate(parts):
    if not part.strip():
        continue
    
    # Check if the previous marker was markdown
    # We need to look at the original content to determine this
    # Find the marker that preceded this part
    preceding_markers = re.findall(r'# %%(?:\s*\[markdown\])?', content)
    
    if i == 0 and not part.strip().startswith('#'):
        # First part before any marker - skip if empty
        continue
    
    # Determine cell type by checking if lines start with #
    lines = part.strip().split('\n')
    is_markdown = all(l.startswith('# ') or l.startswith('#') or l.strip() == '' for l in lines if l.strip())
    
    # Check if this part was preceded by [markdown] marker
    markers_before = content[:content.find(part)]
    last_marker = re.findall(r'# %%(\s*\[markdown\])?', markers_before)
    if last_marker and last_marker[-1] and '[markdown]' in last_marker[-1]:
        is_markdown = True
    
    if is_markdown and all(l.startswith('# ') or l.startswith('#') or l.strip() == '' for l in lines if l.strip()):
        # Convert Python comments to markdown
        md_lines = []
        for l in lines:
            if l.startswith('# '):
                md_lines.append(l[2:])
            elif l.strip() == '#':
                md_lines.append('')
            elif l.strip() == '':
                md_lines.append('')
            else:
                md_lines.append(l)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + '\n' for l in md_lines]
        })
    else:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + '\n' for l in lines]
        })

# Fix: strip trailing newline from last line of each cell
for cell in cells:
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip('\n')

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

with open("submission_notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Created submission_notebook.ipynb with {len(cells)} cells")
print("Cell types:", [c["cell_type"] for c in cells])
