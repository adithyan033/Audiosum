
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# LOAD BART
# ============================================================

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading BART model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("BART model loaded successfully!")


# ============================================================
# SETTINGS
# ============================================================

# Keep below BART's 1024-token input limit
CHUNK_SIZE = 900

# Number of tokens allowed when combining summaries
GROUP_SIZE = 850


# ============================================================
# SPLIT TEXT INTO TOKEN-LIMITED CHUNKS
# ============================================================

def split_into_chunks(text, max_tokens=CHUNK_SIZE):

    sentences = text.split(".")

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
            tokenizer.encode(
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
# SUMMARIZE ONE PIECE OF TEXT
# ============================================================

def summarize_text(
    text,
    max_new_tokens=150,
    min_new_tokens=40
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    outputs = model.generate(

        **inputs,

        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,

        num_beams=4,

        # 1.0 gives a balanced summary length
        length_penalty=1.0,

        early_stopping=True
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# COMBINE SUMMARIES INTO MANAGEABLE GROUPS
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
            tokenizer.encode(
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
# HIERARCHICAL SUMMARIZATION
# ============================================================

def hierarchical_summarize(
    text,
    summary_length="medium"
):

    # --------------------------------------------------------
    # SET SUMMARY LENGTH
    # --------------------------------------------------------

    if summary_length == "short":

        chunk_max = 100
        chunk_min = 30

        final_max = 150
        final_min = 60

    elif summary_length == "detailed":

        chunk_max = 180
        chunk_min = 60

        final_max = 350
        final_min = 150

    else:

# Medium
        chunk_max = 140
        chunk_min = 40

        final_max = 250
        final_min = 100


    # --------------------------------------------------------
    # LEVEL 1
    # --------------------------------------------------------

    chunks = split_into_chunks(text)

    print(
        f"\nText divided into {len(chunks)} chunks."
    )

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
            f"\nCreating summary level "
            f"{level + 1}..."
        )

        groups = group_summaries(summaries)

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

            new_summaries.append(summary)

            print("\nGroup Summary:")
            print(summary)

        summaries = new_summaries

        level += 1


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print("\nGenerating final summary...")

    final_summary = summarize_text(
        summaries[0],
        max_new_tokens=final_max,
        min_new_tokens=final_min
    )

    return final_summary


# ============================================================
# MAIN PROGRAM
# ============================================================

print(
    "\n" + "=" * 60
)

with open("input.txt", "r") as f:
    text = f.read()


# ------------------------------------------------------------
# SELECT SUMMARY LENGTH
# ------------------------------------------------------------

print("\nChoose summary length:")

print("1. Short")
print("2. Medium")
print("3. Detailed")

choice = input(
    "\nEnter your choice (1/2/3): "
)

if choice == "1":

    summary_length = "short"

elif choice == "3":

    summary_length = "detailed"

else:

    summary_length = "medium"


# ------------------------------------------------------------
# GENERATE SUMMARY
# ------------------------------------------------------------

final_summary = hierarchical_summarize(
    text,
    summary_length
)


# ------------------------------------------------------------
# DISPLAY FINAL SUMMARY
# ------------------------------------------------------------

print(
    "\n" + "=" * 60
)

print("\nFINAL SUMMARY:\n")

print(final_summary)

print(
    "\n" + "=" * 60
)

print(
    "\nSummarization completed successfully!"
)