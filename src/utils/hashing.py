import hashlib


def calculate_pdf_hash(uploaded_files):
    """
    Calculate MD5 hash for uploaded PDFs.
    """

    md5 = hashlib.md5()

    for file in uploaded_files:
        md5.update(file.getvalue())

    return md5.hexdigest()