Retrieval-Augmented Generation (RAG)
The RAG design pattern and architecture
RAG lets a model answer questions about data it was never trained on. The pattern:

Ingest: split your corpus into chunks, embed each chunk, store the vectors.
Query: embed the user question, find the k most similar chunks, stuff them into the prompt, ask the model.
It is not magic. It is a search engine bolted onto a generator. Most RAG bugs are search bugs.

Semantic search
Lexical search (BM25, Elasticsearch) matches words. Semantic search matches meaning, via embeddings. "How do I cancel my plan?" retrieves "subscription termination policy". For most production systems you want both: hybrid search, with rerankers on top.

The failure modes of each search type are complementary, which is exactly why hybrid works. Lexical search fails when the user uses different vocabulary than the document: a query about "killing a process" won't find an article about "terminating a job" in a pure keyword system. Semantic search fails when the user uses exact terminology that should match a specific document: serial numbers, product codes, version strings, proper nouns. Embedding similarity doesn't mean string equality.

BM25 is the standard lexical baseline. It scores documents based on term frequency and inverse document frequency with length normalization. It's fast, requires no GPU, and is remarkably competitive with more complex models for many retrieval tasks. Elasticsearch and OpenSearch include it out of the box. For most RAG systems, BM25 plus a dense retriever, fused and reranked, is the right starting point.

Embeddings
Embeddings are dense vectors that place semantically similar texts close together in high-dimensional space. As of early 2026 the strong general options are: Voyage AI voyage-3-large, which on Voyage's own RTEB benchmark (29 retrieval datasets across 8 domains) outperforms OpenAI text-embedding-3-large by 14% and Cohere embed-v4 by 8.2% on NDCG@10; OpenAI text-embedding-3-large (3072 dimensions, supports Matryoshka truncation, $0.13 per million tokens); Cohere embed-v4; Gemini Embedding 2 (multimodal across text, images, video, and audio at $0.15 per million tokens); and BGE-M3 if you self-host.

Embeddings from different models are not compatible. Switching means re-indexing. Pick once, validate, commit.

Vector stores
Two categories.

Libraries: FAISS (in-process, fastest, no metadata), Chroma (embedded, simple), DiskANN. Good for prototypes and small to medium scale.

Databases: Pinecone, Weaviate, Qdrant, Milvus, pgvector, AlloyDB AI, Vertex AI Vector Search. Add metadata filtering, scaling, HA, multi-tenancy.

Pick a library when the corpus fits on a single node and you don't need multi-tenancy. Pick a database when you have multiple writers, need filters, or want to forget about ops.

Storing and searching with Chroma:

python
copy
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
 
client = chromadb.PersistentClient(path="./chroma")
ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
col = client.get_or_create_collection("docs", embedding_function=ef)
 
col.add(documents=[chunk1, chunk2, ...], ids=["c1", "c2", ...])
 
results = col.query(query_texts=["how do I rotate keys?"], n_results=5)
Implementing RAG from scratch
Skeleton, no framework:

python
copy
def embed(text: str) -> list[float]:
    return openai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
 
def retrieve(question: str, k: int = 5) -> list[str]:
    qv = embed(question)
    return vector_store.search(qv, k=k)
 
def answer(question: str) -> str:
    chunks = retrieve(question)
    context = "\n\n".join(chunks)
    prompt = f"Answer the question using only this context:\n\n{context}\n\nQuestion: {question}"
    return llm.complete(prompt)
Everything else is optimization on top of these three functions.

Q&A chatbots: content ingestion, retrieval and generation
Production Q&A bot is RAG plus:

A clean ingestion pipeline (incremental updates, deletes, deduplication).
A retrieval pipeline with hybrid search and a reranker.
Memory of conversation history.
Guardrails on input and output.
Tracing.
The RAG part is one afternoon. The other parts are two months.

Chatbot memory of message history
Two and a half patterns.

Buffer: keep the last N messages verbatim. Cheap, simple, breaks at long conversations.
Summary: as the conversation grows, summarize older turns into a running summary. Loses detail but bounded in size.
Hybrid: keep the last N turns plus a summary of older context. This is what most production chatbots do.
LangGraph does this for you with MessagesState plus a summarization node when message count exceeds a threshold.

Tracing RAG execution
Trace: the user query, the rewritten query, the embedded vector, the retrieved chunks with scores, the reranked chunks, the final prompt, the model output. Without this, debugging a "the bot gave a wrong answer" bug is impossible. LangSmith, Phoenix, and Langfuse all do this with one line of setup. Wire it in on day one.

Advanced RAG
Retrieval algorithms and retrieval optimization
Plain cosine similarity over a single embedding model is the floor, not the ceiling. The path up:

Hybrid search. BM25 + dense, fused with reciprocal rank fusion or weighted scores.
Reranking. Retrieve more candidates than you need, rerank with a cross-encoder (Cohere Rerank, Voyage Rerank, BGE rerankers). Massive quality lift for a small latency cost.
Query expansion. Generate variants of the query, retrieve for each, fuse.
Metadata filtering. Restrict retrieval by source, date, author, language.
Each technique addresses a specific failure mode. Hybrid search fixes vocabulary mismatch. Reranking fixes the gap between "retrieved" and "actually relevant": embedding similarity is a rough proxy, and cross-encoders that compare query and document together are far more precise, just slower. Query expansion fixes queries that are too narrow or phrased in a way the embedder doesn't handle well. Metadata filtering fixes the problem where the right answer exists in your index but is buried under hundreds of older or off-topic documents.

The order matters for implementation. Add hybrid search first: it's the biggest single lift for most corpora and costs almost nothing extra in latency. Add a reranker second: retrieve 50 candidates, rerank, pass the top 5 to the model. Add the others when specific failure patterns emerge in your traces.

Splitting strategies (including HTML-aware splitting)
Recursive character splitting at 512 tokens with 50 to 100 tokens of overlap is the benchmark-validated default for most RAG applications. In FloTorch's February 2026 study comparing seven chunking strategies across 50 academic papers (905,746 tokens, 10+ disciplines, with text-embedding-3-small as the embedder and gemini-2.5-flash-lite as the generator), recursive splitting at 512 tokens scored 69 percent end-to-end accuracy and beat fancier alternatives.

When defaults fail:

Semantic chunking. Split on embedding similarity. Marginal gains, real cost. In the same FloTorch run, semantic chunking produced 43-token average fragments that scored only 54 percent.
HTML / Markdown-aware splitting. Respect headers, lists, code blocks. LangChain's HTMLHeaderTextSplitter and MarkdownHeaderTextSplitter help.
Code-aware splitting. Split on functions, classes, not arbitrary character counts.
Late chunking. Embed full documents, derive chunk vectors via mean pooling. Preserves intra-document context.
There is also a hard ceiling. Bennani et al. (arXiv:2601.14123, École polytechnique) ran a systematic chunking study with SPLADE retrieval and Mistral-8B on Natural Questions and reported, verbatim, that "a 'context cliff' reduces quality beyond ~2.5k tokens". Don't try to beat the model with bigger chunks.

Embedding strategies
The core insight behind all three patterns: the retrieval query and the source document often exist at different levels of abstraction. A user asks a high-level question. The relevant chunk might be a specific paragraph. Direct query-to-chunk matching fails when the vocabulary or abstraction level diverges. Each strategy below addresses that mismatch differently.

Parent/child chunks. Embed small chunks for retrieval, return the larger parent for generation. Best of both worlds.
Document summaries. Embed a summary of the document, plus the chunks. Helps when the user query is high-level.
Hypothetical questions. For each chunk, generate the questions it answers, embed those. The query is a question; matching question-to-question is more reliable than question-to-text.
Parent/child chunking is usually the right default when you want to improve recall without hurting the quality of what the generator sees. Small chunks retrieve precisely; the parent provides context. The hypothetical questions approach works especially well when your source material is answers (documentation, FAQs, knowledge bases) and users naturally phrase queries as questions. Document summary indexing is most useful when your corpus has long heterogeneous documents and users often ask questions that need document-level context rather than a specific passage.

Granular chunk expansion
Retrieve chunks, then pull adjacent chunks for context. The retrieved span widens, the model gets more context, recall goes up. Cheap and effective.

The implementation is straightforward: store each chunk with a reference to its source document and its position within it. When retrieval returns chunk N, expand to include chunks N-1 and N+1 before passing to the generator. If your chunks come from a structured document with headers, you can expand to include everything under the same heading.

This technique is especially valuable for technical documentation, legal text, and anything where a single sentence doesn't make sense without its surrounding context. A retrieved chunk that says "The following exception applies in cases of force majeure" is useless without the sentences before it that establish what rule the exception modifies. Expansion is the cheap fix before reaching for more complex parent/child architectures.

Semi-structured content
Tables, lists, forms. Splitting them as plain text destroys structure. Treat tables specially: extract them as Markdown or JSON, embed a description, put the structured content in the context. Same for code blocks and form fields.

The problem with splitting a table as plain text is that the relationship between column headers and cell values disappears. A chunk reading "Product A 49.99, Product B 39.99, Product C 29.99" means nothing without the header row that tells you what those numbers represent. The header row may have been chunked separately, or not retrieved at all.

The fix is to preserve structure explicitly. For HTML tables, extract as Markdown and store the whole table as a single chunk with a text description of what it contains. For spreadsheet or CSV data, the same: one row per row is fine for storage but not for retrieval. For forms and extraction outputs, use JSON with field names preserved. The description you embed alongside the structured content is what allows a natural-language query to find it; the structured content is what gives the generator something accurate to work with.

Multimodal RAG (RAG beyond text)
Index images, audio, video. Voyage AI multimodal embeddings, Gemini Embedding 2, Cohere Embed v4, ColPali for documents. Query in text, retrieve images. Or describe images during ingestion and retrieve based on the description. The second is simpler and often as good.

Multimodal RAG is more common than it sounds in enterprise contexts. Product catalogs with images, technical documentation with diagrams, support tickets with screenshots, PDFs scanned from paper: all of these appear in real production systems, and all of them break a text-only retrieval pipeline.

Two practical paths. The first is to use a vision model during ingestion to generate text descriptions of images, then embed those descriptions and retrieve them as you would any text chunk. This is slower at ingest time but works with any text embedding model and produces human-readable context for the generator. The second is native multimodal embeddings that place images and text in the same vector space, letting you query in text and retrieve images directly. ColPali is purpose-built for document images: it embeds page images directly using a vision-language model and retrieves them without ever running OCR. Use the description approach as the default unless you need the precision of native multimodal retrieval, or you're working with documents where OCR quality is unreliable.

Question transformations
A question as written is rarely the best query for retrieval. Transformations:

Rewrite-Retrieve-Read. LLM rewrites the question into a search query.
Multiple queries. Generate N variants, retrieve for each, fuse.
Step-back questions. Generate a more general question, retrieve broader context, then narrow.
HyDE. Generate a hypothetical answer, embed that, retrieve documents similar to the answer. Works because answers and source docs share more vocabulary than questions and source docs.
Decomposition. Split a multi-part question into sub-questions, retrieve for each.
Use these together. Most production RAG runs at least multiple queries plus reranking.

Query generation
Sometimes the best retrieval is not vector search:

Self-querying with metadata. LLM extracts filters from the question (author:luca, date>2025-01) and runs a structured query.
Structured SQL. Question to SQL, run on the database, return rows. Best for analytics.
Semantic SQL. SQL with embedding similarity built in (pgvector, AlloyDB AI).
Graph database queries. For knowledge graphs, generate Cypher or SPARQL.
Vector similarity breaks down when the user's intent is inherently structured. "Show me all customers who signed up in January and spent more than $500" is a SQL query in natural-language disguise. Treating it as a semantic search against your knowledge base will return tangentially related documents instead of the rows the user wants.

The discipline here is recognizing query type at routing time. Most questions in a general assistant are semantic. A subset are structured: date ranges, counts, aggregations, filters by known metadata fields. Build explicit classification of query type, route accordingly, and don't try to serve both from the same retrieval backend.

Pick by data shape. SQL beats vector search for structured data. Vector search beats SQL for unstructured.

Chain routing
A single RAG pipeline does not fit every question. Route:

"What is our refund policy?" → policy index.
"How many users signed up last week?" → SQL on analytics DB.
"Summarize this PDF I uploaded". → no retrieval, just summarize.
Use a small classifier model to pick the route. Keep the routes simple.

Retrieval postprocessing
After retrieval, before generation:

Similarity filtering. Drop chunks below a score threshold.
Keyword filtering. Drop chunks that don't match required terms.
Time weighting. Boost recent chunks for time-sensitive questions.
RAG fusion. Run multiple retrievals, fuse with reciprocal rank fusion.
These are cheap, deterministic, and stack nicely. RAG fusion in particular gives a big quality lift for very little code.

From RAG to Agentic RAG
Static RAG is a single retrieve-then-answer pipeline. Agentic RAG is an agent that decides when to retrieve, what to retrieve, whether to retrieve again. A research question takes 4 retrievals, a "hi" takes 0. The agent loops until it has enough context, then answers.

Cost: latency, tokens, complexity. Benefit: handles questions that single-shot RAG can't. Use agentic RAG when your users ask multi-hop or open-ended questions.