import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer, util


# ============================================================
# 1. LOAD BART
# ============================================================

BART_MODEL = "facebook/bart-large-cnn"

print("Loading BART...")

bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)

print("BART loaded successfully!")


# ============================================================
# 2. LOAD SENTENCE TRANSFORMER
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Sentence Transformer loaded successfully!")


# ============================================================
# 3. LOAD NLI MODEL
# ============================================================

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

print("\nLoading NLI model...")

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

print("NLI model loaded successfully!")


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 900
GROUP_SIZE = 850

TOP_K = 5

# Minimum semantic similarity used to select
# relevant source sentences.
SIMILARITY_THRESHOLD = 0.40

# Minimum NLI entailment probability.
ENTAILMENT_THRESHOLD = 0.60


# ============================================================
# 4. SPLIT TEXT INTO SENTENCES
# ============================================================

def split_into_sentences(text):

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
# 5. TOKEN-BASED CHUNKING
# ============================================================

def split_into_chunks(text):

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = len(
            bart_tokenizer.encode(
                sentence,
                add_special_tokens=False
            )
        )

        # If adding this sentence exceeds the limit,
        # save the current chunk.
        if (
            current_tokens + sentence_tokens > CHUNK_SIZE
            and current_chunk
        ):

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Add remaining text
    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


# ============================================================
# 6. BART SUMMARIZATION
# ============================================================

def summarize_text(text):

    inputs = bart_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        summary_ids = bart_model.generate(
            **inputs,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            max_new_tokens=180
        )

    summary = bart_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# 7. SUMMARIZE ALL CHUNKS
# ============================================================

def summarize_chunks(chunks):

    summaries = []

    # print("\nSummarizing chunks...\n")

    for i, chunk in enumerate(chunks):

        # print(
        #     f"Summarizing chunk "
        #     f"{i + 1}/{len(chunks)}..."
        # )

        summary = summarize_text(chunk)

        summaries.append(summary)

    return summaries


# ============================================================
# 8. GROUP SUMMARIES
# ============================================================

def group_summaries(summaries):

    groups = []
    current_group = []
    current_tokens = 0

    for summary in summaries:

        tokens = len(
            bart_tokenizer.encode(
                summary,
                add_special_tokens=False
            )
        )

        if (
            current_tokens + tokens > GROUP_SIZE
            and current_group
        ):

            groups.append(
                " ".join(current_group)
            )

            current_group = []
            current_tokens = 0

        current_group.append(summary)
        current_tokens += tokens

    if current_group:

        groups.append(
            " ".join(current_group)
        )

    return groups


# ============================================================
# 9. HIERARCHICAL SUMMARIZATION
# ============================================================

def hierarchical_summarization(text):

    chunks = split_into_chunks(text)

    print(
        f"\nText divided into "
        f"{len(chunks)} chunks."
    )

    # First level
    summaries = summarize_chunks(chunks)

    # Continue reducing summaries
    while len(summaries) > 1:

        print(
            f"\nReducing "
            f"{len(summaries)} summaries..."
        )

        groups = group_summaries(summaries)

        new_summaries = []

        for group in groups:

            summary = summarize_text(group)

            new_summaries.append(summary)

        summaries = new_summaries

    if summaries:

        return summaries[0]

    return ""


# ============================================================
# 10. FIND RELEVANT SOURCE SENTENCES
# ============================================================

def find_relevant_sources(
    summary_sentence,
    source_sentences
):

    source_embeddings = embedding_model.encode(
        source_sentences,
        convert_to_tensor=True
    )

    summary_embedding = embedding_model.encode(
        summary_sentence,
        convert_to_tensor=True
    )

    # Calculate cosine similarity
    scores = util.cos_sim(
        summary_embedding,
        source_embeddings
    )[0]

    # Get highest scoring sentences
    top_results = torch.topk(
        scores,
        k=min(TOP_K, len(source_sentences))
    )

    relevant_sources = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):

        score_value = float(score)

        if score_value >= SIMILARITY_THRESHOLD:

            relevant_sources.append(
                (
                    source_sentences[int(index)],
                    score_value
                )
            )

    return relevant_sources


# ============================================================
# 11. NLI CHECK
# ============================================================

def check_entailment(
    source_sentence,
    summary_sentence
):

    # Premise = original source
    # Hypothesis = generated summary

    inputs = nli_tokenizer(
        source_sentence,
        summary_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = nli_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    # Find label mapping from model config
    id2label = nli_model.config.id2label

    label_scores = {}

    for index, probability in enumerate(probabilities):

        label = id2label[index].lower()

        label_scores[label] = float(probability)

    return label_scores


# ============================================================
# 12. VERIFY SUMMARY SENTENCE
# ============================================================

def verify_summary_sentence(
    summary_sentence,
    source_sentences
):

    relevant_sources = find_relevant_sources(
        summary_sentence,
        source_sentences
    )

    if not relevant_sources:

        return {
            "status": "UNSUPPORTED",
            "source": None,
            "similarity": 0,
            "entailment": 0,
            "contradiction": 0,
            "neutral": 0
        }

    best_result = None

    for source, similarity in relevant_sources:

        nli_scores = check_entailment(
            source,
            summary_sentence
        )

        entailment = 0
        contradiction = 0
        neutral = 0

        for label, score in nli_scores.items():

            if "entail" in label:
                entailment = score

            elif "contrad" in label:
                contradiction = score

            elif "neutral" in label:
                neutral = score

        result = {
            "source": source,
            "similarity": similarity,
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral
        }

        # Keep candidate with highest entailment
        if (
            best_result is None
            or entailment > best_result["entailment"]
        ):

            best_result = result

    # Determine status
    if best_result["entailment"] >= ENTAILMENT_THRESHOLD:

        status = "SUPPORTED"

    elif best_result["contradiction"] >= 0.60:

        status = "POSSIBLE CONTRADICTION"

    else:

        status = "POTENTIALLY UNSUPPORTED"

    best_result["status"] = status

    return best_result


# ============================================================
# 13. VERIFY COMPLETE SUMMARY
# ============================================================

def verify_summary(
    summary,
    original_text
):

    source_sentences = split_into_sentences(
        original_text
    )

    summary_sentences = split_into_sentences(
        summary
    )

    print("\n")
    print("=" * 70)
    print("FACTUALITY VERIFICATION")
    print("=" * 70)

    results = []

    for i, sentence in enumerate(
        summary_sentences
    ):

        print(
            f"\nChecking sentence "
            f"{i + 1}/{len(summary_sentences)}..."
        )

        result = verify_summary_sentence(
            sentence,
            source_sentences
        )

        results.append(
            (sentence, result)
        )

    return results


# ============================================================
# 14. NEW FUNCTION FOR COMPLETE SUMMARIZATION
# ============================================================
#
# This is the important change.
#
# Whisper will send its transcript directly here.
#
# Whisper transcript
#        ↓
# hierarchical_summarization()
#        ↓
# final summary
#        ↓
# verify_summary()
#
# ============================================================

def summarize_transcript(text):

    if not text or not text.strip():
        print("ERROR: Empty transcript.")
        return ""

    # Generate summary
    final_summary = hierarchical_summarization(text)

    # Print the exact generated summary
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(final_summary)

    # Verify summary
    verification_results = verify_summary(
        final_summary,
        text
    )

    # Return the EXACT same summary
    return final_summary
# ============================================================
# 15. OPTIONAL TEST
# ============================================================
#
# This section is only for testing summarizer.py separately.
#
# It is NOT needed when pipeline.py is being used.
#
# ============================================================

if __name__ == "__main__":

    try:

        with open(
            "input.txt",
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except FileNotFoundError:

        print(
            "ERROR: input.txt not found."
        )

        print(
            "\nWhen using the complete audio pipeline,"
        )

        print(
            "run pipeline.py instead."
        )

    else:

        if not text.strip():

            print(
                "ERROR: input.txt is empty."
            )

        else:

            summarize_transcript(text)