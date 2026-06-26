import fitz
from langchain_core.documents import Document


def load_pdf(pdf_path: str):
    """
    Load a PDF and return LangChain Document objects.
    """

    documents = []

    pdf = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf):

        text = page.get_text()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num + 1,
                },
            )
        )

    pdf.close()

    return documents