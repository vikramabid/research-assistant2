from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents, faiss_retriever):

        self.documents = documents

        self.faiss = faiss_retriever

        tokenized_docs = [
            doc.page_content.split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    def invoke(self, query, k=15):

        # -------- BM25 --------

        tokens = query.split()

        bm25_docs = self.bm25.get_top_n(
            tokens,
            self.documents,
            n=k
        )

        # -------- FAISS --------

        faiss_docs = self.faiss.invoke(query)

        # -------- Merge --------

        merged = []

        seen = set()

        for doc in bm25_docs + faiss_docs:

            text = doc.page_content

            if text not in seen:

                seen.add(text)

                merged.append(doc)

        return merged[:k]