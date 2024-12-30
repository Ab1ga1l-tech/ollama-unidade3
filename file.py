import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings

# Define os documentos
documents = ["Sim.", "Professor.", "Salada.", "Oi."]

# Define o modelo de embeddings
try:
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
except Exception as e:
    print(f"Erro ao inicializar os embeddings: {e}")
    raise

# Gera embeddings para os documentos
try:
    document_embeddings = embeddings.embed_documents(documents)
    embedding_size = len(document_embeddings[0])
    print(f"Tamanho dos embeddings: {embedding_size}")
except Exception as e:
    print(f"Erro ao gerar embeddings para os documentos: {e}")
    raise

# Consulta e similaridade
query = "Uma profissão?"
try:
    query_embedding = embeddings.embed_query(query)
    similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]
    most_similar_index = np.argmax(similarity_scores)
    most_similar_document = documents[most_similar_index]

    print(f"Documento mais similar à consulta '{query}': {most_similar_document}")
except Exception as e:
    print(f"Erro ao calcular similaridade: {e}")
    raise


# Exibindo informações sobre os embeddings
print(f"Embeddings gerados para o texto: '{text}'")
print(f"Tamanho do vetor de embeddings: {len(embeddings)}")
print(f"Primeiros 5 valores do vetor: {embeddings[:5]}")

