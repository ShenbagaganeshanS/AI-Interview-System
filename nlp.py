from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_answer(user_answer: str, reference_answer: str):
    if not user_answer.strip():
        return "POOR", 0.0

    embeddings = model.encode([user_answer, reference_answer])

    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    print(f"Semantic Similarity: {similarity:.3f}")

    if similarity > 0.7:
        return "GOOD", similarity
    elif similarity > 0.4:
        return "AVERAGE", similarity
    else:
        return "POOR", similarity