
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==========================================
# LOAD BART MODEL
# ==========================================

model_name = "facebook/bart-large-cnn"

print("Loading BART model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")


# ==========================================
# FUNCTION 1: SPLIT TEXT INTO CHUNKS
# ==========================================

def split_into_chunks(text, max_tokens=900):

    # Simple sentence splitting
    sentences = text.split(".")

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # Remove unnecessary spaces
        sentence = sentence.strip()

        # Ignore empty sentences
        if not sentence:
            continue

        # Add the full stop back
        sentence = sentence + "."

        # Test the new chunk
        test_chunk = current_chunk + " " + sentence

        # Count tokens
        token_count = len(
            tokenizer.encode(
                test_chunk,
                add_special_tokens=True
            )
        )

        # If token limit is exceeded
        if token_count > max_tokens:

            # Save the current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start a new chunk
            current_chunk = sentence

        else:

            # Continue adding sentences
            current_chunk = test_chunk

    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ==========================================
# FUNCTION 2: SUMMARIZE EACH CHUNK
# ==========================================

def summarize_chunk(chunk):

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    outputs = model.generate(

        **inputs,

        # Length of each chunk summary
        max_new_tokens=150,
        min_new_tokens=50,

        # Beam search
        num_beams=4,

        # Encourage a reasonably detailed summary
        length_penalty=1.5,

        early_stopping=True
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ==========================================
# FUNCTION 3: GENERATE FINAL SUMMARY
# ==========================================

def generate_final_summary(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    outputs = model.generate(

        **inputs,

        # Final summary can be longer
        max_new_tokens=220,
        min_new_tokens=80,

        num_beams=4,

        # Allow a more detailed summary
        length_penalty=1.2,

        early_stopping=True
    )

    final_summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return final_summary


# ==========================================
# MAIN PROGRAM
# ==========================================

print("\n" + "=" * 60)

text = input("\nEnter your text:\n")


# ------------------------------------------
# STEP 1: SPLIT INTO CHUNKS
# ------------------------------------------

chunks = split_into_chunks(text)

print("\n" + "=" * 60)

print(f"\nText divided into {len(chunks)} chunks.")


# ------------------------------------------
# STEP 2: SUMMARIZE EACH CHUNK
# ------------------------------------------

chunk_summaries = []

for i, chunk in enumerate(chunks):

    print("\n" + "-" * 60)

    print(f"\nSummarizing Chunk {i + 1} of {len(chunks)}...")

    summary = summarize_chunk(chunk)

    chunk_summaries.append(summary)

    print("\nCHUNK SUMMARY:")

    print(summary)


# ------------------------------------------
# STEP 3: COMBINE ALL SUMMARIES
# ------------------------------------------

combined_summary = " ".join(chunk_summaries)

print("\n" + "=" * 60)

print("\nCOMBINED SUMMARY:\n")

print(combined_summary)

# ------------------------------------------
# STEP 4: GENERATE FINAL SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)

print("\nGenerating Final Summary... Please wait.")


# Check if combined summary is within BART limit
combined_tokens = len(
    tokenizer.encode(
        combined_summary,
        add_special_tokens=True
    )
)


print(f"\nCombined summary token count: {combined_tokens}")


# If combined summary is small enough
if combined_tokens <= 1024:

    final_summary = generate_final_summary(combined_summary)

else:

    # If the combined summary is still too long,
    # split it again into smaller chunks

    print("\nCombined summary is too long.")
    print("Performing second-level summarization...")

    second_level_chunks = split_into_chunks(
        combined_summary,
        max_tokens=900
    )

    second_level_summaries = []

    for i, chunk in enumerate(second_level_chunks):

        print(
            f"\nSummarizing second-level chunk "
            f"{i + 1} of {len(second_level_chunks)}..."
        )

        summary = summarize_chunk(chunk)

        second_level_summaries.append(summary)

    # Combine second-level summaries
    second_combined_summary = " ".join(second_level_summaries)

    # Generate final summary
    final_summary = generate_final_summary(
        second_combined_summary
    )


# ------------------------------------------
# DISPLAY FINAL RESULT
# ------------------------------------------

print("\n" + "=" * 60)

print("\nFINAL SUMMARY:\n")

print(final_summary)

print("\n" + "=" * 60)

print("\nSummarization completed successfully!")