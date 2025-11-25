import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = "model/flan_t5/"

def load_flan_model():
    """
    Memuat model FLAN-T5 dari folder lokal.
    Mengembalikan tokenizer & model.
    """
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    return tokenizer, model


def summarize_text(text, tokenizer, model, max_input=1024, max_output=250):
    """
    Melakukan summarization menggunakan FLAN-T5

    Parameter:
        text (str): teks panjang yang akan diringkas
        tokenizer: tokenizer FLAN-T5
        model: model FLAN-T5
        max_input (int): batas maksimal panjang input text
        max_output (int): panjang maksimal ringkasan

    Return:
        summary (str)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input,
        truncation=True
    ).to(device)

    output = model.generate(
        inputs,
        max_length=max_output,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary
