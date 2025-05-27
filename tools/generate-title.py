# Overview:
# This script generates a URL-friendly title from a given string.
# It converts the string to lowercase, replaces spaces with hyphens,
# and removes special characters except hyphens.
# generate_title.py
# Usage:
# python3 generate-title.py "How to use Python for data analysis"
# Example:
# print(generate_title("How to use Python for data analysis"))
# Output: "how-to-use-python-for-data-analysis"import re

import re

def generate_title(title: str) -> str:
    # Convert to lowercase
    title = title.lower()
    # Replace spaces with hyphens
    title = title.replace(" ", "-")
    # Remove special characters except hyphens
    title = re.sub(r'[^a-z0-9-]', '', title)
    return title

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate-title.py <title>")
        sys.exit(1)
    title = sys.argv[1]
    print(generate_title(title))
    sys.exit(0)
