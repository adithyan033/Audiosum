from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import re


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

bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    BART_MODEL
)

print("BART model loaded successfully!")


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Sentence Transformer loaded successfully!")


# ============================================================
# SETTINGS
# ============================================================

# Keep below BART's 1024-token input limit
CHUNK_SIZE = 900

# Maximum size when combining summaries
GROUP_SIZE = 850

# Similarity threshold for factual support
SIMILARITY_THRESHOLD = 0.55


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_into_chunks(
    text,
    max_tokens=CHUNK_SIZE
):

    # Split at ., ! or ?
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

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
# SUMMARIZE ONE TEXT
# ============================================================

def summarize_text(
    text,
    max_new_tokens=150,
    min_new_tokens=40
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

            no_repeat_ngram_size=3,

            length_penalty=1.0,

            early_stopping=True
        )


    summary = bart_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return summary


# ============================================================
# GROUP SUMMARIES
# ============================================================

def group_summaries(
    summaries,
    max_tokens=GROUP_SIZE
):

    groups = []

    current_group = ""

    for summary in summaries:

        test_group = (
            current_group + " " + summary
        ).strip()

        token_count = len(
            bart_tokenizer.encode(
                test_group,
                add_special_tokens=True
            )
        )

        if token_count > max_tokens:

            if current_group:

                groups.append(
                    current_group.strip()
                )

            current_group = summary

        else:

            current_group = test_group


    if current_group:

        groups.append(
            current_group.strip()
        )


    return groups


# ============================================================
# HIERARCHICAL BART SUMMARIZATION
# ============================================================

def hierarchical_summarize(
    text,
    summary_length="medium"
):

    # --------------------------------------------------------
    # SUMMARY LENGTH SETTINGS
    # --------------------------------------------------------

    if summary_length == "short":

        chunk_max = 100
        chunk_min = 30

    elif summary_length == "detailed":

        chunk_max = 180
        chunk_min = 60

    else:

        # Medium

        chunk_max = 140
        chunk_min = 40


    # --------------------------------------------------------
    # LEVEL 1
    # --------------------------------------------------------

    chunks = split_into_chunks(text)

    print(
        f"\nText divided into "
        f"{len(chunks)} chunks."
    )


    if not chunks:

        return ""


    summaries = []


    for i, chunk in enumerate(chunks):

        print(
            f"\nSummarizing chunk "
            f"{i + 1}/{len(chunks)}..."
        )


        summary = summarize_text(
            chunk,
            max_new_tokens=chunk_max,
            min_new_tokens=chunk_min
        )


        summaries.append(summary)


        print("\nChunk Summary:")

        print(summary)


    # --------------------------------------------------------
    # HIERARCHICAL REDUCTION
    # --------------------------------------------------------

    level = 1


    while len(summaries) > 1:

        print(
            "\n" + "=" * 60
        )

        print(
            f"\nSummary reduction level "
            f"{level + 1}"
        )


        groups = group_summaries(
            summaries
        )


        print(
            f"Created {len(groups)} groups."
        )


        new_summaries = []


        for i, group in enumerate(groups):

            print(
                f"\nSummarizing group "
                f"{i + 1}/{len(groups)}..."
            )


            summary = summarize_text(
                group,
                max_new_tokens=chunk_max,
                min_new_tokens=chunk_min
            )


            new_summaries.append(
                summary
            )


            print("\nGroup Summary:")

            print(summary)


        summaries = new_summaries

        level += 1


    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT SUMMARIZE summaries[0] AGAIN
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "\nHierarchical summarization completed."
    )


    return summaries[0]


# ============================================================
# SPLIT SUMMARY INTO SENTENCES
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# FACTUAL SUPPORT CHECK
# ============================================================

def check_factual_support(
    summary,
    original_text
):

    print(
        "\n" + "=" * 60
    )

    print(
        "\nChecking factual support..."
    )


    # --------------------------------------------------------
    # Split original text into sentences
    # --------------------------------------------------------

    source_sentences = split_sentences(
        original_text
    )


    # --------------------------------------------------------
    # Split final summary
    # --------------------------------------------------------

    summary_sentences = split_sentences(
        summary
    )


    if not source_sentences:

        return []


    # --------------------------------------------------------
    # Create source embeddings
    # --------------------------------------------------------

    print(
        "\nCreating source embeddings..."
    )


    source_embeddings = embedding_model.encode(
        source_sentences,
        convert_to_tensor=True
    )


    verification_results = []


    # --------------------------------------------------------
    # Check each summary sentence
    # --------------------------------------------------------

    for i, summary_sentence in enumerate(
        summary_sentences
    ):

        print(
            f"\nChecking summary sentence "
            f"{i + 1}/{len(summary_sentences)}..."
        )


        summary_embedding = (
            embedding_model.encode(
                summary_sentence,
                convert_to_tensor=True
            )
        )


        # Compare summary sentence
        # with every source sentence

        scores = torch.nn.functional.cosine_similarity(
            summary_embedding,
            source_embeddings
        )


        best_score = torch.max(scores).item()

        best_index = torch.argmax(scores).item()

        supporting_sentence = (
            source_sentences[best_index]
        )


        is_supported = (
            best_score >= SIMILARITY_THRESHOLD
        )


        verification_results.append({

            "summary_sentence":
                summary_sentence,

            "similarity":
                best_score,

            "supporting_sentence":
                supporting_sentence,

            "supported":
                is_supported
        })


    return verification_results


# ============================================================
# DISPLAY FACTUALITY REPORT
# ============================================================

def display_verification_report(
    results
):

    print(
        "\n" + "=" * 60
    )

    print(
        "\nFACTUAL SUPPORT REPORT"
    )

    print(
        "=" * 60
    )


    for i, result in enumerate(
        results
    ):

        print(
            f"\nSummary Sentence {i + 1}:"
        )

        print(
            result["summary_sentence"]
        )


        print(
            f"\nSimilarity Score: "
            f"{result['similarity']:.4f}"
        )


        print(
            "\nMost similar source sentence:"
        )

        print(
            result["supporting_sentence"]
        )


        if result["supported"]:

            print(
                "\nSTATUS: ✓ Potentially supported"
            )

        else:

            print(
                "\nSTATUS: ⚠ Potentially unsupported"
            )


        print(
            "-" * 60
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "\nReading input.txt..."
)


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


# ============================================================
# SELECT SUMMARY LENGTH
# ============================================================

print(
    "\nChoose summary length:"
)

print(
    "1. Short"
)

print(
    "2. Medium"
)

print(
    "3. Detailed"
)


choice = input(
    "\nEnter your choice (1/2/3): "
)


if choice == "1":

    summary_length = "short"

elif choice == "3":

    summary_length = "detailed"

else:

    summary_length = "medium"


# ============================================================
# GENERATE SUMMARY
# ============================================================

final_summary = hierarchical_summarize(
    text,
    summary_length
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "\nFINAL SUMMARY:\n"
)

print(
    final_summary
)


# ============================================================
# FACTUALITY VERIFICATION
# ============================================================

verification_results = (
    check_factual_support(
        final_summary,
        text
    )
)


display_verification_report(
    verification_results
)


# ============================================================
# COMPLETED
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "\nBART summarization + "
    "factual support verification completed!"
)