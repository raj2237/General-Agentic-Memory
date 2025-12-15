"""
Codebase Cleanup Script

This script will:
1. Remove test files
2. Remove redundant documentation
3. Clean up excessive comments in code files
4. Remove Chinese comments and translate if needed
"""

import os
import re
from pathlib import Path

# Files to delete
FILES_TO_DELETE = [
    "backend/test.py",
    "backend/test_improvements.py",
    "backend/test_retriever_fix.py",
    "IMPROVEMENTS_SUMMARY.md",
    "QUICK_START_IMPROVEMENTS.md",
]

# Files to clean (remove excessive comments)
FILES_TO_CLEAN = [
    "backend/api/chat.py",
    "backend/api/documents.py",
    "backend/core/gam_manager.py",
    "backend/utils/graph_builder.py",
    "gam/agents/research_agent.py",
]

def remove_files():
    """Remove unnecessary files"""
    for file_path in FILES_TO_DELETE:
        full_path = Path(file_path)
        if full_path.exists():
            full_path.unlink()
            print(f"✅ Deleted: {file_path}")
        else:
            print(f"⏭️  Skipped (not found): {file_path}")

def clean_comments(content: str) -> str:
    """Remove excessive comments while keeping docstrings"""
    lines = content.split('\n')
    cleaned_lines = []
    in_docstring = False
    docstring_char = None
    
    for line in lines:
        stripped = line.strip()
        
        # Track docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_char = stripped[:3]
                cleaned_lines.append(line)
            elif stripped.endswith(docstring_char):
                in_docstring = False
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
            continue
        
        if in_docstring:
            cleaned_lines.append(line)
            continue
        
        # Remove Chinese comments
        if re.search(r'[\u4e00-\u9fff]', line):
            # Skip lines with Chinese characters (comments)
            continue
        
        # Remove excessive inline comments (keep code)
        if '#' in line and not stripped.startswith('#'):
            # Keep code, remove comment if it's too verbose
            code_part = line.split('#')[0].rstrip()
            if code_part:
                cleaned_lines.append(code_part)
        elif not stripped.startswith('#'):
            # Keep non-comment lines
            cleaned_lines.append(line)
        elif 'TODO' in line or 'FIXME' in line or 'NOTE' in line:
            # Keep important comments
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def main():
    print("🧹 Starting codebase cleanup...")
    print("\n" + "="*60)
    
    # Step 1: Remove files
    print("\n📁 Removing unnecessary files...")
    remove_files()
    
    print("\n" + "="*60)
    print("\n✅ Cleanup complete!")
    print("\nNote: Code file cleanup should be done manually to preserve logic.")
    print("Use the cleanup plan in .cleanup_plan.md for guidance.")

if __name__ == "__main__":
    main()
