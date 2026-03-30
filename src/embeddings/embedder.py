from unsloth import FastSentenceTransformer


model = FastSentenceTransformer.from_pretrained(
    model_name="unsloth/all-MiniLM-L6-v2",
    max_seq_length=512,
    full_finetuning=False,
)


def generate_embeddings(texts, is_query):
    """
    Generate embeddings for a list of texts
    """
    if is_query:
        return model.encode_query(texts, show_progress_bar=True)
    return model.encode_document(texts, show_progress_bar=True)


def build_candidate_text(candidate):
    past_roles_str = "\n".join(
        [
            f"- {role['role']}: {role['summary']}"
            for role in candidate.get("past_roles", [])
        ]
    )
    return f"""
    Skills: {', '.join(candidate.get('skills', []))}
    Experience: {candidate.get('total_experience', 0)} years
    Role: {candidate.get('current_role', '')}
    Past Roles: {past_roles_str}
    """
