from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch


# ============================================================
# MODELS
# ============================================================

BART_MODEL = "facebook/bart-large-cnn"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD BART
# ============================================================

print("Loading BART model...")

bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)

print("BART loaded successfully!")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded successfully!")


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 500

# Number of relevant chunks retrieved
TOP_K = 5

# Maximum input tokens given to BART
BART_INPUT_LIMIT = 900


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_into_chunks(text, max_tokens=CHUNK_SIZE):

    sentences = text.replace("!", ".") \
                    .replace("?", ".") \
                    .split(".")

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence = sentence + "."

        test_chunk = (
            current_chunk + " " + sentence
        ).strip()

        token_count = len(
            bart_tokenizer.encode(
                test_chunk,
                add_special_tokens=True
            )
        )

        if token_count > max_tokens:

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = sentence

        else:

            current_chunk = test_chunk

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    print("\nCreating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_tensor=True
    )

    return embeddings


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(chunks, embeddings, query, top_k=TOP_K):

    print("\nRetrieving relevant information...")

    # Convert query into an embedding
    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True
    )

    # Calculate similarity
    scores = torch.nn.functional.cosine_similarity(
        query_embedding,
        embeddings
    )

    # Get highest scoring chunks
    top_results = torch.topk(
        scores,
        k=min(top_k, len(chunks))
    )

    retrieved_chunks = []

    print("\nRetrieved chunks:")

    for index in top_results.indices:

        index = index.item()

        print(
            f"\nChunk {index + 1} "
            f"(score: {scores[index].item():.4f})"
        )

        print(chunks[index])

        retrieved_chunks.append(
            chunks[index]
        )

    return retrieved_chunks


# ============================================================
# PREPARE CONTEXT FOR BART
# ============================================================

def prepare_context(chunks):

    context = ""

    for chunk in chunks:

        test_context = (
            context + "\n" + chunk
        ).strip()

        token_count = len(
            bart_tokenizer.encode(
                test_context,
                add_special_tokens=True
            )
        )

        if token_count > BART_INPUT_LIMIT:
            break

        context = test_context

    return context


# ============================================================
# BART SUMMARIZATION
# ============================================================

def summarize_with_bart(
    text,
    max_new_tokens=250,
    min_new_tokens=80
):

    inputs = bart_tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    with torch.no_grad():

        outputs = bart_model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,

            num_beams=4,

            # Prevent excessive repetition
            no_repeat_ngram_size=3,

            # Balanced generation
            length_penalty=1.0,

            early_stopping=True
        )

    summary = bart_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# RAG + BART PIPELINE
# ============================================================

def rag_bart_summarize(text):

    # --------------------------------------------------------
    # STEP 1: SPLIT TEXT
    # --------------------------------------------------------

    chunks = split_into_chunks(text)

    print(
        f"\nText divided into "
        f"{len(chunks)} chunks."
    )

    if not chunks:

        print("\nERROR: No text was found.")

        return ""


    # --------------------------------------------------------
    # STEP 2: CREATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = create_embeddings(chunks)


    # --------------------------------------------------------
    # STEP 3: RETRIEVAL
    # --------------------------------------------------------

    # For summarization, we use a general query
    query = (
        "Important information, main ideas, "
        "key facts and important events in the text"
    )

    retrieved_chunks = retrieve_chunks(
        chunks,
        embeddings,
        query
    )


    # --------------------------------------------------------
    # STEP 4: PREPARE RETRIEVED CONTEXT
    # --------------------------------------------------------

    context = prepare_context(
        retrieved_chunks
    )

    print(
        "\n" + "=" * 60
    )

    print("\nContext sent to BART:")

    print(context)


    # --------------------------------------------------------
    # STEP 5: BART SUMMARIZATION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print("\nGenerating summary...")

    summary = summarize_with_bart(
        context,
        max_new_tokens=250,
        min_new_tokens=80
    )


    return summary


# ============================================================
# MAIN PROGRAM
# ============================================================

print(
    "\n" + "=" * 60
)

# ------------------------------------------------------------
# READ INPUT
# ------------------------------------------------------------

with open(
    "input.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


if not text.strip():

    print(
        "\nERROR: input.txt is empty!"
    )

    exit()


print(
    f"\nInput text length: "
    f"{len(text)} characters"
)


# ------------------------------------------------------------
# RUN RAG + BART
# ------------------------------------------------------------

final_summary = rag_bart_summarize(
    text
)


# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "\nFINAL SUMMARY:\n"
)

print(final_summary)

print(
    "\n" + "=" * 60
)

print(
    "\nRAG + BART summarization completed!"
)