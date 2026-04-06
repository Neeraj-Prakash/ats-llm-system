from unsloth import FastLanguageModel


def get_model(seq_length=2048, load_in_4bit=True):

    print("🔄 Loading model...")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(_model)
    print("✅ Model loaded!")

    return _model, _tokenizer
