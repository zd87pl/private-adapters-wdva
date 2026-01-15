# WDVA: Secure & Private AI for Everyone

> **Your data. Your model. Your privacy. Zero compromise.**

Weight-Delta Vault Adapters (WDVA) enables **personalized AI without sacrificing privacy**. Train a model on your documents, encrypt it, and run it entirely on your device—no cloud, no data sharing, no compromise.

## 🔐 The Promise

**Traditional AI:**
- ❌ Share your data with cloud providers
- ❌ Models trained on your data stored on servers you don't control
- ❌ No way to delete your data once it's trained
- ❌ Privacy vs. Personalization tradeoff

**WDVA Approach:**
- ✅ Your data stays encrypted, always
- ✅ Train once, run anywhere—even offline
- ✅ Cryptographic "right to be forgotten" (delete the key = delete the model)
- ✅ Personalization without privacy compromise

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run arXiv example
python examples/arxiv_demo.py --paper-id 2502.13171

# Or use the simple demo
python examples/simple_demo.py
```

## 📚 What is WDVA?

**Weight-Delta Vault Adapters** is a privacy-preserving AI personalization technique:

1. **Train** a small adapter (DoRA) on your documents
2. **Encrypt** the adapter with military-grade cryptography (XChaCha20-Poly1305)
3. **Store** encrypted adapter anywhere (cloud, local, USB drive)
4. **Decrypt** and load ephemerally (in-memory only) when needed
5. **Delete** instantly by destroying the encryption key

### Key Properties

- 🔒 **Zero-knowledge**: Server never sees your data or decrypted model
- 💾 **Portable**: Encrypted adapter is small (~20MB) and portable
- ⚡ **Fast**: Load adapter in milliseconds, switch between users instantly
- 🗑️ **Deletable**: Cryptographic deletion—destroy key = model is gone forever
- 🏠 **Local-first**: Run entirely on your device, no internet required

## 🎯 Use Cases

### For Consumers

- **Personal Knowledge Base**: Train on your notes, documents, emails - create your own AI that knows everything you know
- **Private Research**: Query research papers, books, and articles without sharing them with any cloud service
- **Family Documents**: Organize and search family records, recipes, medical history privately
- **Creative Writing**: Train on your writing style for personalized assistance without exposing unpublished work
- **Financial Planning**: Query your financial documents without exposing sensitive data to third parties

### For Enterprises

- **HIPAA-Compliant Healthcare**: Build personalized patient assistants that keep PHI (Protected Health Information) encrypted and local
- **Legal Discovery**: Train on case files for private legal research - attorney-client privilege preserved
- **Financial Services (SOC 2/PCI)**: Customer data never leaves your infrastructure, enabling AI without compliance violations
- **Defense & Government**: Classified document analysis with zero data exfiltration risk
- **Pharmaceutical R&D**: Query proprietary research without exposing IP to cloud providers
- **HR & Recruiting**: Search confidential employee records and candidate data privately
- **Customer Support**: Per-customer AI assistants trained on their specific history and context

### Why Enterprises Choose WDVA

| Requirement | Traditional Cloud AI | WDVA |
|-------------|---------------------|------|
| Data Residency | Data leaves premises | Data never leaves |
| Compliance (GDPR, HIPAA, SOC 2) | Complex DPAs required | Built-in compliance |
| Data Deletion | Uncertain, may persist in training | Cryptographic guarantee |
| Audit Trail | Provider-dependent | Full local control |
| Offline Operation | Requires internet | Works completely offline |
| Per-User Isolation | Shared models | Isolated encrypted adapters |

## 📖 Examples

### arXiv Paper Assistant

Train on a research paper and query it privately:

```python
from wdva import WDVA
from examples.arxiv_demo import download_and_train

# Download paper and train adapter
adapter_path, key = download_and_train("2502.13171")

# Create WDVA instance
wdva = WDVA(adapter_path=adapter_path, encryption_key=key)

# Query privately (runs entirely locally)
response = wdva.query("What is the main contribution of this paper?")
print(response)
```

### Simple Document Training

```python
from wdva import WDVA

# Train on your documents
wdva = WDVA()
wdva.train(
    documents=["doc1.txt", "doc2.pdf"],
    model_name="TinyLlama-1.1B-Chat"
)

# Query your personal AI
answer = wdva.query("What did I write about privacy?")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Your Documents                  │
│    (PDFs, Notes, Emails, etc.)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      DoRA Training (Local)              │
│    Generates small adapter (~20MB)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Encryption (XChaCha20-Poly1305)      │
│    Creates encrypted adapter blob       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Encrypted Adapter Storage            │
│  (Cloud, Local, USB - doesn't matter)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Ephemeral Loading (Your Device)      │
│  Decrypt → Load → Query → Delete        │
│      (All in memory, never on disk)     │
└─────────────────────────────────────────┘
```

## 🔗 LangChain Integration

WDVA integrates seamlessly with the LangChain ecosystem for building privacy-preserving AI applications.

```bash
# Install with LangChain support
pip install wdva[langchain]       # Core LangChain
pip install wdva[langchain-full]  # Full RAG support with embeddings
```

### Quick Example: Private RAG

```python
from wdva.langchain_integration import (
    WDVAChatModel,
    WDVADocumentLoader,
    WDVAEmbeddings,
    create_wdva_chain
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load your private documents (locally)
loader = WDVADocumentLoader(file_paths=["confidential.pdf", "notes.txt"])
docs = loader.load()

# 2. Create local embeddings (no cloud API!)
embeddings = WDVAEmbeddings()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Query with your encrypted adapter
chain = create_wdva_chain(
    adapter_path="my_adapter.wdva",
    encryption_key="your-64-char-hex-key",
    retriever=vectorstore.as_retriever()
)

response = chain.invoke({"query": "What are the key findings?"})
```

### Available Components

| Component | Description |
|-----------|-------------|
| `WDVALLM` | Standard LLM interface for LangChain chains |
| `WDVAChatModel` | Chat model with message history support |
| `WDVADocumentLoader` | Load PDF, TXT, HTML, JSON documents |
| `WDVAEmbeddings` | Privacy-preserving local embeddings |
| `WDVACallbackHandler` | Privacy-aware logging and monitoring |

See [docs/LANGCHAIN_REVIEW.md](docs/LANGCHAIN_REVIEW.md) for the complete integration guide.

## 🔧 Extending WDVA

See [docs/EXTENDING.md](docs/EXTENDING.md) for:
- Custom data sources
- Different model backends
- Custom encryption schemes
- LangChain integration patterns
- FastAPI deployment

## 📚 Documentation

- [CONCEPT.md](docs/CONCEPT.md) - Understanding WDVA
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical deep dive
- [EXTENDING.md](docs/EXTENDING.md) - How to extend WDVA
- [LANGCHAIN_REVIEW.md](docs/LANGCHAIN_REVIEW.md) - LangChain integration guide

## 🤝 Contributing

This is a reference implementation demonstrating the WDVA concept. Feel free to:
- Use it as a starting point for your own projects
- Extend it with new features
- Share improvements

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

Apache 2.0 provides:
- ✅ Explicit patent grants (important for AI/ML projects)
- ✅ Patent retaliation protection
- ✅ Still very permissive (like MIT)
- ✅ Common in AI/ML open source projects

## 🙏 Acknowledgments

WDVA builds on:
- **DoRA** (Decomposed Low-Rank Adaptation) for efficient fine-tuning
- **XChaCha20-Poly1305** for authenticated encryption
- **Small Language Models** (TinyLlama, Llama-3.2-1B) for local inference

---

**Privacy is not a feature—it's a fundamental right. WDVA makes it possible.**

