# Document vectorizer

This service uses [langchain](https://python.langchain.com/docs/get_started/introduction) to vectorize documents into a [FAISS](https://faiss.ai/index.html) vectorstore.

The embdedding model used is [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) available in [Huggingface](https://huggingface.co/).

These vectorstores can then be used to do similarity searches in this vector space.
This process is used in RAG (Retrieval Augmented Generation) to augment the generation process of a LLM with information from the document.
