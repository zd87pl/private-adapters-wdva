# Extending WDVA

This guide shows how to extend WDVA for your specific use case.

## Custom Data Sources

### Example: Email Loader

```python
from wdva import WDVA

class EmailLoader:
    def load(self, email_path: str) -> str:
        # Load and parse email
        # Return text content
        pass

# Use in training
wdva = WDVA()
emails = EmailLoader().load_all("~/emails")
wdva.train(documents=emails)
```

### Example: Database Loader

```python
class DatabaseLoader:
    def __init__(self, connection_string: str):
        self.conn = connect(connection_string)
    
    def load(self, query: str) -> str:
        # Query database
        # Format as text
        # Return
        pass
```

## Custom Models

### Using Different Base Models

```python
from wdva import WDVA

# Use Llama-3.2-1B instead of TinyLlama
wdva = WDVA(model_name="meta-llama/Llama-3.2-1B-Instruct")

# Use Qwen2.5-1.5B
wdva = WDVA(model_name="Qwen/Qwen2.5-1.5B-Instruct")
```

### Custom Model Backend

```python
from wdva.inference import LocalInference

class CustomInference(LocalInference):
    def load_model(self):
        # Custom model loading logic
        pass
```

## Custom Encryption

### Example: Hardware Security Module (HSM)

```python
from wdva.crypto import EncryptedAdapter

class HSMEncryptedAdapter(EncryptedAdapter):
    def _derive_key(self, salt: bytes, info: bytes) -> bytes:
        # Use HSM for key derivation
        return hsm.derive_key(self.encryption_key, salt, info)
```

## Cloud Storage Integration

### Example: S3 Storage

```python
import boto3
from wdva import WDVA

class S3Storage:
    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
    
    def save(self, adapter_path: str, key: str):
        self.s3.upload_file(adapter_path, self.bucket, key)
    
    def load(self, key: str) -> str:
        local_path = f"/tmp/{key}"
        self.s3.download_file(self.bucket, key, local_path)
        return local_path

# Usage
storage = S3Storage("my-adapters-bucket")
adapter_path, encryption_key = wdva.train(documents=["doc.pdf"])

# Upload encrypted adapter (safe - it's encrypted!)
storage.save(adapter_path, "user123/adapter.wdva")

# Download and query
local_path = storage.load("user123/adapter.wdva")
wdva.load(local_path, encryption_key)
response = wdva.query("What is this about?")
```

## Multi-User Support

### Example: User Manager

```python
from wdva import WDVA
import json

class UserManager:
    def __init__(self, storage_dir: str = "./users"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def create_user(self, user_id: str) -> WDVA:
        """Create a new user with their own adapter."""
        user_dir = self.storage_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        return WDVA()
    
    def save_user_key(self, user_id: str, encryption_key: str):
        """Save encryption key (encrypted with user's master password)."""
        key_file = self.storage_dir / user_id / "key.enc"
        # Encrypt key with user's master password
        # Save encrypted key
        pass
    
    def load_user(self, user_id: str, master_password: str) -> WDVA:
        """Load user's adapter."""
        adapter_path = self.storage_dir / user_id / "adapter.wdva"
        encryption_key = self._decrypt_user_key(user_id, master_password)
        
        return WDVA(adapter_path=str(adapter_path), encryption_key=encryption_key)

# Usage
manager = UserManager()

# Create user
user_wdva = manager.create_user("alice")
adapter_path, key = user_wdva.train(documents=["alice_doc.pdf"])
manager.save_user_key("alice", key)

# Load user
alice_wdva = manager.load_user("alice", master_password="alice_password")
response = alice_wdva.query("What did I write?")
```

## Custom Training Pipeline

### Example: Advanced Training

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

def train_dora_advanced(
    model_name: str,
    dataset: list,
    output_dir: str
) -> tuple[str, str]:
    """Advanced DoRA training with full control."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure DoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply DoRA
    model = get_peft_model(model, lora_config)
    
    # Train (simplified - full implementation would include proper dataset handling)
    # ... training code ...
    
    # Save adapter
    adapter_path = f"{output_dir}/adapter"
    model.save_pretrained(adapter_path)
    
    # Encrypt
    from wdva.crypto import EncryptedAdapter
    import secrets
    
    encryption_key = secrets.token_bytes(32)
    adapter = EncryptedAdapter(encryption_key)
    
    # Load adapter weights and encrypt
    # ... encryption code ...
    
    return adapter_path, encryption_key.hex()
```

## Integration Examples

### LangChain Integration

```python
from langchain.llms.base import LLM
from wdva import WDVA

class WDVALLM(LLM):
    def __init__(self, wdva: WDVA):
        self.wdva = wdva
    
    def _call(self, prompt: str, stop: list = None) -> str:
        return self.wdva.query(prompt)
    
    @property
    def _llm_type(self) -> str:
        return "wdva"

# Use with LangChain
wdva = WDVA(adapter_path="my_adapter.wdva", encryption_key="...")
llm = WDVALLM(wdva)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=my_prompt)
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from wdva import WDVA

app = FastAPI()

# Store adapters in memory (in production, use proper storage)
adapters = {}

@app.post("/train")
async def train_adapter(user_id: str, documents: list[str]):
    wdva = WDVA()
    adapter_path, key = wdva.train(documents)
    adapters[user_id] = {"path": adapter_path, "key": key}
    return {"adapter_id": user_id, "status": "trained"}

@app.post("/query")
async def query(user_id: str, prompt: str):
    if user_id not in adapters:
        raise HTTPException(404, "Adapter not found")
    
    adapter_info = adapters[user_id]
    wdva = WDVA(
        adapter_path=adapter_info["path"],
        encryption_key=adapter_info["key"]
    )
    
    response = wdva.query(prompt)
    return {"response": response}
```

## Best Practices

1. **Key Management**: Store encryption keys securely (password manager, HSM, etc.)
2. **Backup**: Backup encrypted adapters (they're safe to store anywhere)
3. **Versioning**: Version your adapters for rollback capability
4. **Testing**: Test encryption/decryption before deploying
5. **Monitoring**: Monitor inference performance and memory usage

## Contributing

Found a bug or have an improvement? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

