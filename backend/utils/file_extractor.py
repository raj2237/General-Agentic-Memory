import io
import os
import PyPDF2

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Fast text extraction - PyPDF2 first for speed!
    Falls back to unstructured only if needed.
    """
    file_ext = os.path.splitext(filename.lower())[1]

    try:
        # FAST PATH: Use PyPDF2 for PDFs (10x faster than unstructured)
        if file_ext == ".pdf":
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                if text.strip():
                    return text
            except Exception as pdf_error:
                print(f"[EXTRACT] PyPDF2 failed, trying fallback: {pdf_error}")
        
        # FAST PATH: Plain text files
        if file_ext in [".txt", ".md", ".csv"]:
            try:
                return file_content.decode('utf-8')
            except:
                return file_content.decode('latin-1')
        
        # SLOW FALLBACK: Use unstructured for other formats (DOCX, etc.)
        try:
            from unstructured.partition.auto import partition
            
            elements = partition(file=io.BytesIO(file_content))
            text = "\n\n".join([
                str(el.text)
                for el in elements
                if hasattr(el, "text") and el.category not in ["Image", "Table"]
            ])
            
            if text.strip():
                return text
        except ImportError:
            print("[EXTRACT] Warning: unstructured not available, skipping")
        except Exception as e:
            print(f"[EXTRACT] Unstructured failed: {e}")

        return "No readable text found."

    except Exception as e:
        raise ValueError(f"Failed to extract text from {filename}: {str(e)}")
