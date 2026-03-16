from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import os
import zipfile
from io import BytesIO

settings = get_settings()

api_description = """
This service uses langchain to vectorize documents into a FAISS vectorstore.
You can chat with your vectorized documents using the
Swiss AI Center's [chatbot](https://chatbot.swiss-ai-center.ch/).
"""
api_summary = """
This service uses langchain to vectorize documents into a FAISS vectorstore.
"""
api_title = "Document Vectorizer API."
version = "0.0.1"


class MyService(Service):
    """
    This service uses langchain to vectorize documents into a FAISS vectorstore.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger
    _embedding_model: object
    _vector_path: object

    def __init__(self):
        super().__init__(
            name="Document Vectorizer",
            slug="document-vectorizer",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="document",
                    type=[
                        FieldDescriptionType.APPLICATION_PDF,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_ZIP]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DOCUMENT_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.DOCUMENT_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/document-vectorizer/",
        )
        self._logger = get_logger(settings)
        self._embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_path = "./vectorstore"

    def process(self, data):
        raw = data["document"].data

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

        # raw content as PDF temp file
        with open("temp.pdf", "wb") as f:
            f.write(raw)

        loader = PyMuPDFLoader("temp.pdf")
        doc = loader.load_and_split(text_splitter)
        vectorstore = FAISS.from_documents(
            documents=doc,
            embedding=self._embedding_model,
        )
        if os.path.exists(self._vector_path):
            shutil.rmtree(self._vector_path)
        vectorstore.save_local(self._vector_path)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self._vector_path):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file),
                            os.path.join(self._vector_path, ".."),
                        ),
                    )
        return {
            "result": TaskData(
                data=zip_buffer.getvalue(), type=FieldDescriptionType.APPLICATION_ZIP
            )
        }
