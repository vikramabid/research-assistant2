import fitz
from langchain_core.documents import Document
from src.logger.logger import logger

def load_pdf(pdf_path, source_name):
    documents = []

    pdf = fitz.open(pdf_path)
    logger.info(f"Loading PDF: {source_name}")
    for page_num in range(len(pdf)):

        page = pdf.load_page(page_num)

        text = page.get_text()

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num + 1
                    }
                )
            )

    pdf.close()
    logger.info(f"Loaded {len(documents)} pages from {source_name}")
    return documents