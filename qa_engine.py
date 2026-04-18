import pypdf
from collections import deque
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-3.5-turbo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-V2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_DOCS = 4
MEMORY_WINDOW = 10

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rephrase "
            "the follow-up into a single self-contained question that can be "
            "answered without the history. If it is already standalone, return "
            "it unchanged. Output ONLY the rephrased question, nothing else."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise document assistant. Answer the user's question "
            "using ONLY the information in the retrieved document excerpts below. "
            "If the answer is not contained in the excerpts, say: "
            '"I couldn\'t find that information in the document." '
            "Never fabricate facts or reference outside knowledge.\n\n"
            "Retrieved excerpts:\n"
            "──────────────────\n"
            "{context}\n"
            "──────────────────"
        ),
        ("human", "{question}"),
    ]
)

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )

class QAEngine:
    def __init__(self, api_key: str) -> None:
        self._api_key: str | None = api_key
        self._vectorstore: FAISS | None = None
        self._retriever = None
        self._llm: ChatOpenAI | None = None
        self._history: deque[BaseMessage] = deque(maxlen=MEMORY_WINDOW)
        self._embeddings: OpenAIEmbeddings | None = None

    def ingest(self, pdf_path: Path) -> None:
        docs = self._load_pdf(pdf_path)
        chunks = self._split(docs)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # self._embeddings = OpenAIEmbeddings(
        #     api_key=self._api_key,
        #     base_url=OPENROUTER_BASE_URL,
        # )
        self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_DOCS},
        )
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=self._api_key,
            temperature=0,
            max_tokens=512,
        )
        self._history.clear()

    @staticmethod
    def _load_pdf(path: Path) -> list[Document]:
        # loader = PyPDFLoader(str(path))
        # pages = loader.load()
        reader = pypdf.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1, "source": str(path)},
                    )
                )
        if not pages:
            raise ValueError("No extractable text found. The PDF may be scanned or image-only.")
        return pages

    @staticmethod
    def _split(docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")
        return chunks

    def _condense_question(self, question: str) -> str:
        if not self._history:
            return question
        condense_chain = CONDENSE_PROMPT | self._llm | StrOutputParser()
        return condense_chain.invoke(
            {"question": question, "chat_history": list(self._history)}
        )

    def _build_qa_chain(self):
        retrieve = RunnablePassthrough.assign(
            docs=RunnableLambda(lambda x: self._retriever.invoke(x["question"]))
        )
        format_context = RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(x["docs"]))
        )
        answer_generation_chain = (
            RunnableLambda(
                lambda x: {"context": x["context"], "question": x["original_question"]}
            )
            | QA_PROMPT
            | self._llm
            | StrOutputParser()
        )
        return retrieve | format_context | RunnablePassthrough.assign(answer=answer_generation_chain)

    def ask(self, question: str) -> dict:
        if self._retriever is None:
            raise RuntimeError(
                "No document has been ingested yet. Call ingest() first."
            )

        standalone_question = self._condense_question(question)
        chain = self._build_qa_chain()
        result = chain.invoke(
            {"question": standalone_question, "original_question": question}
        )

        answer: str = result["answer"]
        source_docs: list[Document] = result.get("docs", [])

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        return {"answer": answer, "source_documents": source_docs}