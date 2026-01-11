## RAG (src/tracerag/rag/naive.py)

Purpose: assemble context from retrieved chunks and call a chat model.
Retrieval is intentionally separate (see `documentation/retrieval.md`).

### Types

`RagResponse`
- `answer` (str): final chat completion content.
- `chunks` (List[Chunk]): retrieved chunks used to build context.
- `context` (str): full context string passed to the model.
- `messages` (List[Dict[str, str]]): chat messages sent to the model.

### Functions

`build_openai_chat_fn(client, model, temperature=0.2) -> ChatFn`
- Input:
  - `client` (OpenAI): initialized OpenAI client.
  - `model` (str): chat model name.
  - `temperature` (float): sampling temperature.
- Output: `ChatFn` callable that accepts `messages` and returns a string.

`build_context_text(chunks, include_sources=True, separator="---") -> str`
- Input:
  - `chunks` (Iterable[Chunk]): retrieval results.
  - `include_sources` (bool): include source header per chunk.
  - `separator` (str): section separator.
- Output: context string for the model.

`build_rag_messages(system_prompt, user_query, context, answer_instruction="Return only the answer.") -> List[Dict[str, str]]`
- Input:
  - `system_prompt` (str)
  - `user_query` (str)
  - `context` (str)
  - `answer_instruction` (str)
- Output: list of chat messages (OpenAI-compatible).

`naive_rag(user_query, retriever, chat_fn, system_prompt, top_k=8, context_builder=build_context_text, answer_instruction="Return only the answer.") -> RagResponse`
- Input:
  - `user_query` (str)
  - `retriever` (Callable[[str, Optional[int]], List[Chunk]]): retrieval function.
  - `chat_fn` (ChatFn)
  - `system_prompt` (str)
  - `top_k` (int): number of chunks to request.
  - `context_builder` (Callable): builds context string from chunks.
  - `answer_instruction` (str)
- Output: `RagResponse` containing answer + inputs used.

### How to create and run a new RAG pipeline

You compose retrieval + chat + prompt. Example:

```python
from openai import OpenAI
from qdrant_client import QdrantClient

from tracerag.rag import build_openai_chat_fn, naive_rag
from tracerag.retrieval import (
    QdrantRetrievalConfig,
    build_openai_embedding_fn,
    build_qdrant_retriever,
)

client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

embed_fn = build_openai_embedding_fn(client, model="text-embedding-3-small")
retriever = build_qdrant_retriever(
    qdrant=qdrant,
    embedding_fn=embed_fn,
    config=QdrantRetrievalConfig(collection_name="catalog_embeddings"),
)

chat_fn = build_openai_chat_fn(client, model="gpt-4.1-mini", temperature=0.2)
system_prompt = "You are a Cisco IOS XR network engineer."

resp = naive_rag(
    "Generate telemetry configuration for IOS XR about BGP.",
    retriever=lambda q, k: retriever(q, top_k=k),
    chat_fn=chat_fn,
    system_prompt=system_prompt,
    top_k=8,
)

print(resp.answer)
```

### Inputs and outputs at a glance

- Input query: string.
- Retrieval output: list of `Chunk` objects.
- Context string: assembled from `Chunk.text`.
- Output: `RagResponse` with `answer`, `context`, `messages`, and `chunks`.
