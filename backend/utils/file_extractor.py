from unstructured.partition.auto import partition
import io
import os
import PyPDF2

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    file_ext = os.path.splitext(filename.lower())[1]

    try:
        elements = partition(
            file=io.BytesIO(file_content)
        )

        text = "\n\n".join([
            str(el.text)
            for el in elements
            if hasattr(el, "text") and el.category not in ["Image", "Table"]
        ])

        if text.strip():
            return text

        
        if file_ext == ".pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return "\n".join([page.extract_text() or "" for page in reader.pages])

        return text or "No readable text found."

    except Exception as e:
        raise ValueError(f"Failed to extract text from {filename}: {str(e)}")
