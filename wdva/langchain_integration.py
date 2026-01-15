"""
LangChain Integration for WDVA - Privacy-Preserving AI within the LangChain Ecosystem.

This module provides comprehensive LangChain compatibility for WDVA, enabling:
- Custom LLM and ChatModel interfaces
- Document loaders for private data
- Retrieval-based QA chains
- Callbacks for observability
- Memory management for conversations

Example:
    >>> from wdva.langchain_integration import WDVAChatModel, WDVADocumentLoader
    >>> from langchain.chains import ConversationalRetrievalChain
    >>>
    >>> chat = WDVAChatModel(adapter_path="adapter.wdva", encryption_key="...")
    >>> chain = ConversationalRetrievalChain.from_llm(chat, retriever=retriever)
    >>> response = chain.invoke({"question": "What is in my documents?"})

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Conditional Imports with Graceful Degradation
# =============================================================================

_langchain_available = False
_langchain_community_available = False
_langchain_core_available = False

try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        AIMessageChunk,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.documents import Document
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.embeddings import Embeddings
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

    _langchain_core_available = True
    _langchain_available = True
except ImportError:
    logger.debug("langchain-core not available, LangChain integration disabled")


# =============================================================================
# WDVA LLM Interface (Modern LangChain Core)
# =============================================================================

if _langchain_core_available:

    class WDVALLM(LLM):
        """
        LangChain LLM interface for WDVA.

        This provides a standard LangChain LLM that uses WDVA's encrypted
        adapters for privacy-preserving inference.

        Example:
            >>> from wdva.langchain_integration import WDVALLM
            >>> from langchain.chains import LLMChain
            >>> from langchain_core.prompts import PromptTemplate
            >>>
            >>> llm = WDVALLM(
            ...     adapter_path="my_adapter.wdva",
            ...     encryption_key="your-64-char-hex-key"
            ... )
            >>>
            >>> prompt = PromptTemplate.from_template("Question: {question}\\nAnswer:")
            >>> chain = prompt | llm
            >>> response = chain.invoke({"question": "What is this about?"})

        Attributes:
            adapter_path: Path to encrypted WDVA adapter file
            encryption_key: Hex-encoded encryption key (64 characters)
            model_name: Base model name (default: TinyLlama-1.1B)
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature (0.0-1.0)
        """

        adapter_path: str
        encryption_key: str
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        max_tokens: int = 512
        temperature: float = 0.7

        # Private instance cache (not a Pydantic field)
        _wdva_instance: Optional[Any] = None

        model_config = {"arbitrary_types_allowed": True}

        def __init__(
            self,
            adapter_path: str,
            encryption_key: str,
            model_name: Optional[str] = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            **kwargs: Any,
        ) -> None:
            """
            Initialize WDVA LLM.

            Args:
                adapter_path: Path to encrypted adapter file
                encryption_key: Hex-encoded encryption key
                model_name: Base model name (optional)
                max_tokens: Maximum tokens per generation
                temperature: Sampling temperature
                **kwargs: Additional LangChain LLM parameters
            """
            super().__init__(
                adapter_path=adapter_path,
                encryption_key=encryption_key,
                model_name=model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            # Initialize private attribute after super().__init__
            object.__setattr__(self, '_wdva_instance', None)

        def _get_wdva(self) -> Any:
            """Lazy-load WDVA instance."""
            if self._wdva_instance is None:
                from wdva import WDVA
                self._wdva_instance = WDVA(
                    adapter_path=self.adapter_path,
                    encryption_key=self.encryption_key,
                    model_name=self.model_name,
                )
            return self._wdva_instance

        @property
        def _llm_type(self) -> str:
            """Return identifier for this LLM type."""
            return "wdva"

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Return parameters that identify this LLM."""
            return {
                "adapter_path": self.adapter_path,
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """
            Execute the LLM on the given prompt.

            Args:
                prompt: The prompt to generate from
                stop: Optional list of stop sequences
                run_manager: Callback manager for the run
                **kwargs: Additional generation parameters

            Returns:
                Generated text response
            """
            wdva = self._get_wdva()

            response = wdva.query(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )

            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]

            return response

        @property
        def is_loaded(self) -> bool:
            """Check if adapter is loaded."""
            if self._wdva_instance is None:
                return False
            return self._wdva_instance.is_loaded

        def unload(self) -> None:
            """Unload adapter and free memory."""
            if self._wdva_instance is not None:
                self._wdva_instance.unload()

        def delete(self) -> None:
            """Cryptographically delete the adapter."""
            if self._wdva_instance is not None:
                self._wdva_instance.delete()
                self._wdva_instance = None


    # =========================================================================
    # WDVA Chat Model Interface
    # =========================================================================

    class WDVAChatModel(BaseChatModel):
        """
        LangChain Chat Model interface for WDVA.

        Provides chat-style interactions with message history support,
        compatible with all LangChain chat-based chains and agents.

        Example:
            >>> from wdva.langchain_integration import WDVAChatModel
            >>> from langchain_core.messages import HumanMessage, SystemMessage
            >>>
            >>> chat = WDVAChatModel(
            ...     adapter_path="adapter.wdva",
            ...     encryption_key="your-key-here"
            ... )
            >>>
            >>> messages = [
            ...     SystemMessage(content="You are a helpful assistant."),
            ...     HumanMessage(content="What is in my documents?")
            ... ]
            >>> response = chat.invoke(messages)

        Attributes:
            adapter_path: Path to encrypted WDVA adapter file
            encryption_key: Hex-encoded encryption key
            model_name: Base model name
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Default system prompt (optional)
        """

        adapter_path: str
        encryption_key: str
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        max_tokens: int = 512
        temperature: float = 0.7
        system_prompt: Optional[str] = None

        # Private instance cache (not a Pydantic field)
        _wdva_instance: Optional[Any] = None

        model_config = {"arbitrary_types_allowed": True}

        def __init__(
            self,
            adapter_path: str,
            encryption_key: str,
            model_name: Optional[str] = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            system_prompt: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            """
            Initialize WDVA Chat Model.

            Args:
                adapter_path: Path to encrypted adapter file
                encryption_key: Hex-encoded encryption key
                model_name: Base model name (optional)
                max_tokens: Maximum tokens per generation
                temperature: Sampling temperature
                system_prompt: Default system prompt
                **kwargs: Additional parameters
            """
            super().__init__(
                adapter_path=adapter_path,
                encryption_key=encryption_key,
                model_name=model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
            # Initialize private attribute after super().__init__
            object.__setattr__(self, '_wdva_instance', None)

        def _get_wdva(self) -> Any:
            """Lazy-load WDVA instance."""
            if self._wdva_instance is None:
                from wdva import WDVA
                self._wdva_instance = WDVA(
                    adapter_path=self.adapter_path,
                    encryption_key=self.encryption_key,
                    model_name=self.model_name,
                )
            return self._wdva_instance

        @property
        def _llm_type(self) -> str:
            """Return identifier for this chat model type."""
            return "wdva-chat"

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Return parameters that identify this model."""
            return {
                "adapter_path": self.adapter_path,
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

        def _format_messages(self, messages: List[BaseMessage]) -> str:
            """
            Format chat messages into a prompt string.

            Uses a simple chat format compatible with most models.
            Override this method for model-specific formatting.

            Args:
                messages: List of chat messages

            Returns:
                Formatted prompt string
            """
            formatted_parts = []

            # Add default system prompt if not in messages
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system and self.system_prompt:
                formatted_parts.append(f"System: {self.system_prompt}")

            for message in messages:
                if isinstance(message, SystemMessage):
                    formatted_parts.append(f"System: {message.content}")
                elif isinstance(message, HumanMessage):
                    formatted_parts.append(f"User: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_parts.append(f"Assistant: {message.content}")
                else:
                    # Generic fallback
                    formatted_parts.append(f"{message.type}: {message.content}")

            # Add assistant prefix for generation
            formatted_parts.append("Assistant:")

            return "\n\n".join(formatted_parts)

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """
            Generate a response from chat messages.

            Args:
                messages: List of chat messages
                stop: Optional stop sequences
                run_manager: Callback manager
                **kwargs: Additional parameters

            Returns:
                ChatResult with generated message
            """
            wdva = self._get_wdva()

            # Format messages to prompt
            prompt = self._format_messages(messages)

            # Generate response
            response_text = wdva.query(
                prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )

            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]

            # Create chat generation
            message = AIMessage(content=response_text.strip())
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """
            Async generation (runs sync version in executor).

            Note: WDVA inference is CPU/GPU-bound, so true async
            would require additional infrastructure.
            """
            return self._generate(messages, stop, run_manager, **kwargs)

        def unload(self) -> None:
            """Unload adapter and free memory."""
            if self._wdva_instance is not None:
                self._wdva_instance.unload()

        def delete(self) -> None:
            """Cryptographically delete the adapter."""
            if self._wdva_instance is not None:
                self._wdva_instance.delete()
                self._wdva_instance = None


    # =========================================================================
    # WDVA Document Loader
    # =========================================================================

    class WDVADocumentLoader(BaseLoader):
        """
        LangChain Document Loader for WDVA-compatible documents.

        Loads documents from various sources while preserving metadata
        for use with WDVA training and retrieval.

        Example:
            >>> from wdva.langchain_integration import WDVADocumentLoader
            >>>
            >>> loader = WDVADocumentLoader(
            ...     file_paths=["doc1.pdf", "doc2.txt"],
            ...     metadata={"source": "private_docs"}
            ... )
            >>> documents = loader.load()
            >>>
            >>> # Use with text splitter for chunking
            >>> from langchain.text_splitter import RecursiveCharacterTextSplitter
            >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            >>> chunks = splitter.split_documents(documents)

        Attributes:
            file_paths: List of file paths to load
            metadata: Optional metadata to add to all documents
            encoding: Text encoding (default: utf-8)
        """

        def __init__(
            self,
            file_paths: List[Union[str, Path]],
            metadata: Optional[Dict[str, Any]] = None,
            encoding: str = "utf-8",
        ) -> None:
            """
            Initialize document loader.

            Args:
                file_paths: List of file paths to load
                metadata: Optional metadata for all documents
                encoding: Text encoding
            """
            self.file_paths = [Path(p) for p in file_paths]
            self.metadata = metadata or {}
            self.encoding = encoding

        def load(self) -> List[Document]:
            """
            Load documents from files.

            Returns:
                List of LangChain Document objects
            """
            documents = []

            for file_path in self.file_paths:
                if not file_path.exists():
                    logger.warning("File not found: %s", file_path)
                    continue

                try:
                    content = self._load_file(file_path)

                    doc_metadata = {
                        **self.metadata,
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                    }

                    documents.append(Document(
                        page_content=content,
                        metadata=doc_metadata,
                    ))

                except Exception as e:
                    logger.error("Failed to load %s: %s", file_path, e)

            return documents

        def _load_file(self, file_path: Path) -> str:
            """
            Load content from a single file.

            Supports: .txt, .md, .pdf, .html, .json

            Args:
                file_path: Path to file

            Returns:
                File content as string
            """
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                return self._load_pdf(file_path)
            elif suffix in (".html", ".htm"):
                return self._load_html(file_path)
            elif suffix == ".json":
                return self._load_json(file_path)
            else:
                # Default: plain text
                return file_path.read_text(encoding=self.encoding)

        def _load_pdf(self, file_path: Path) -> str:
            """Load PDF file."""
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(file_path))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            except ImportError:
                raise ImportError(
                    "PyPDF2 required for PDF loading. "
                    "Install with: pip install PyPDF2"
                )

        def _load_html(self, file_path: Path) -> str:
            """Load HTML file and extract text."""
            try:
                from bs4 import BeautifulSoup
                html = file_path.read_text(encoding=self.encoding)
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n", strip=True)
            except ImportError:
                # Fallback: basic HTML stripping
                import re
                html = file_path.read_text(encoding=self.encoding)
                return re.sub(r'<[^>]+>', '', html)

        def _load_json(self, file_path: Path) -> str:
            """Load JSON file and convert to readable text."""
            import json
            data = json.loads(file_path.read_text(encoding=self.encoding))
            return json.dumps(data, indent=2)

        def lazy_load(self) -> Iterator[Document]:
            """Lazy load documents one at a time."""
            for file_path in self.file_paths:
                if not file_path.exists():
                    continue

                try:
                    content = self._load_file(file_path)
                    doc_metadata = {
                        **self.metadata,
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                    }
                    yield Document(page_content=content, metadata=doc_metadata)
                except Exception as e:
                    logger.error("Failed to load %s: %s", file_path, e)


    # =========================================================================
    # WDVA Embeddings (Local, Private)
    # =========================================================================

    class WDVAEmbeddings(Embeddings):
        """
        Privacy-preserving embeddings using local models.

        Unlike cloud-based embeddings, these run entirely locally,
        ensuring your document content never leaves your device.

        Example:
            >>> from wdva.langchain_integration import WDVAEmbeddings
            >>> from langchain_community.vectorstores import FAISS
            >>>
            >>> embeddings = WDVAEmbeddings()
            >>> vectorstore = FAISS.from_documents(documents, embeddings)
            >>> retriever = vectorstore.as_retriever()

        Attributes:
            model_name: Embedding model name
            device: Device to run on (auto-detected)
        """

        def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: Optional[str] = None,
        ) -> None:
            """
            Initialize local embeddings.

            Args:
                model_name: Sentence transformer model name
                device: Device to use (auto-detected if None)
            """
            self.model_name = model_name
            self._model: Any = None
            self._device = device or self._detect_device()

        def _detect_device(self) -> str:
            """Detect best available device."""
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        def _get_model(self) -> Any:
            """Lazy-load the embedding model."""
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self._device
                    )
                except ImportError:
                    raise ImportError(
                        "sentence-transformers required for embeddings. "
                        "Install with: pip install sentence-transformers"
                    )
            return self._model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """
            Embed a list of documents.

            Args:
                texts: List of text strings to embed

            Returns:
                List of embedding vectors
            """
            model = self._get_model()
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        def embed_query(self, text: str) -> List[float]:
            """
            Embed a single query.

            Args:
                text: Query text to embed

            Returns:
                Embedding vector
            """
            model = self._get_model()
            embedding = model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()


    # =========================================================================
    # WDVA Callback Handler
    # =========================================================================

    class WDVACallbackHandler(BaseCallbackHandler):
        """
        Callback handler for WDVA operations.

        Provides privacy-preserving logging and monitoring without
        sending data to external services.

        Example:
            >>> from wdva.langchain_integration import WDVAChatModel, WDVACallbackHandler
            >>>
            >>> callback = WDVACallbackHandler(log_prompts=False)  # Don't log sensitive data
            >>> chat = WDVAChatModel(
            ...     adapter_path="adapter.wdva",
            ...     encryption_key="...",
            ...     callbacks=[callback]
            ... )

        Attributes:
            log_prompts: Whether to log prompts (default: False for privacy)
            log_responses: Whether to log responses (default: False for privacy)
            log_tokens: Whether to log token counts (default: True)
        """

        def __init__(
            self,
            log_prompts: bool = False,
            log_responses: bool = False,
            log_tokens: bool = True,
        ) -> None:
            """
            Initialize callback handler.

            Args:
                log_prompts: Whether to log prompts
                log_responses: Whether to log responses
                log_tokens: Whether to log token counts
            """
            self.log_prompts = log_prompts
            self.log_responses = log_responses
            self.log_tokens = log_tokens

            self.total_prompts = 0
            self.total_completions = 0
            self.total_tokens_estimated = 0

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            **kwargs: Any,
        ) -> None:
            """Called when LLM starts running."""
            self.total_prompts += len(prompts)

            if self.log_prompts:
                for prompt in prompts:
                    logger.info("WDVA Prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
            else:
                logger.debug("WDVA LLM started with %d prompts", len(prompts))

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            """Called when LLM ends running."""
            self.total_completions += 1

            if self.log_responses and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = gen.text if hasattr(gen, 'text') else str(gen)
                        logger.info("WDVA Response: %s", text[:100] + "..." if len(text) > 100 else text)

            # Estimate tokens (rough: ~4 chars per token)
            if response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = gen.text if hasattr(gen, 'text') else str(gen)
                        self.total_tokens_estimated += len(text) // 4

            if self.log_tokens:
                logger.debug("WDVA completion finished, ~%d total tokens", self.total_tokens_estimated)

        def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
            """Called when LLM errors."""
            logger.error("WDVA LLM error: %s", str(error))

        def get_stats(self) -> Dict[str, int]:
            """Get usage statistics."""
            return {
                "total_prompts": self.total_prompts,
                "total_completions": self.total_completions,
                "total_tokens_estimated": self.total_tokens_estimated,
            }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_wdva_chain(
    adapter_path: str,
    encryption_key: str,
    chain_type: str = "stuff",
    **kwargs: Any,
) -> Any:
    """
    Create a LangChain QA chain with WDVA.

    Args:
        adapter_path: Path to encrypted adapter
        encryption_key: Encryption key
        chain_type: Chain type ("stuff", "map_reduce", "refine")
        **kwargs: Additional chain parameters

    Returns:
        Configured LangChain chain

    Example:
        >>> chain = create_wdva_chain(
        ...     adapter_path="adapter.wdva",
        ...     encryption_key="...",
        ...     retriever=my_retriever
        ... )
        >>> response = chain.invoke({"query": "What is this about?"})
    """
    if not _langchain_core_available:
        raise ImportError("LangChain required. Install with: pip install langchain-core")

    chat = WDVAChatModel(
        adapter_path=adapter_path,
        encryption_key=encryption_key,
    )

    # Try to create retrieval chain if retriever provided
    retriever = kwargs.pop("retriever", None)

    if retriever is not None:
        try:
            from langchain.chains import RetrievalQA
            return RetrievalQA.from_chain_type(
                llm=chat,
                chain_type=chain_type,
                retriever=retriever,
                **kwargs
            )
        except ImportError:
            pass

    # Return just the chat model if no retriever
    return chat


def create_conversational_chain(
    adapter_path: str,
    encryption_key: str,
    retriever: Any = None,
    memory: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Create a conversational chain with WDVA and memory.

    Args:
        adapter_path: Path to encrypted adapter
        encryption_key: Encryption key
        retriever: Optional retriever for RAG
        memory: Optional conversation memory
        **kwargs: Additional parameters

    Returns:
        Conversational chain

    Example:
        >>> from langchain.memory import ConversationBufferMemory
        >>>
        >>> memory = ConversationBufferMemory(return_messages=True)
        >>> chain = create_conversational_chain(
        ...     adapter_path="adapter.wdva",
        ...     encryption_key="...",
        ...     retriever=vectorstore.as_retriever(),
        ...     memory=memory
        ... )
        >>> response = chain.invoke({"question": "What did we discuss?"})
    """
    if not _langchain_core_available:
        raise ImportError("LangChain required. Install with: pip install langchain-core")

    chat = WDVAChatModel(
        adapter_path=adapter_path,
        encryption_key=encryption_key,
    )

    if retriever is not None:
        try:
            from langchain.chains import ConversationalRetrievalChain

            if memory is None:
                from langchain.memory import ConversationBufferMemory
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )

            return ConversationalRetrievalChain.from_llm(
                llm=chat,
                retriever=retriever,
                memory=memory,
                **kwargs
            )
        except ImportError:
            logger.warning("langchain package not available for chains")

    return chat


# =============================================================================
# Module Exports
# =============================================================================

# Only export classes if LangChain is available
if _langchain_core_available:
    __all__ = [
        "WDVALLM",
        "WDVAChatModel",
        "WDVADocumentLoader",
        "WDVAEmbeddings",
        "WDVACallbackHandler",
        "create_wdva_chain",
        "create_conversational_chain",
    ]
else:
    __all__ = []

    # Provide helpful error for missing LangChain
    def _langchain_not_available(*args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install wdva[langchain]"
        )

    # Create placeholder classes that raise helpful errors
    WDVALLM = _langchain_not_available  # type: ignore
    WDVAChatModel = _langchain_not_available  # type: ignore
    WDVADocumentLoader = _langchain_not_available  # type: ignore
    WDVAEmbeddings = _langchain_not_available  # type: ignore
    WDVACallbackHandler = _langchain_not_available  # type: ignore
    create_wdva_chain = _langchain_not_available  # type: ignore
    create_conversational_chain = _langchain_not_available  # type: ignore
