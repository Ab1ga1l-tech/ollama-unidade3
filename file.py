import requests
import numpy as np

# URL da API do Ollama
url = "http://localhost:11500/v1/models/llama2/embeddings"


# Texto simples para gerar embeddings
text = "Esta é uma frase de exemplo para gerar embeddings."

# Função para gerar embeddings com Ollama
def get_embeddings(text):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    # Requisição para a API do Ollama
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # A resposta contém os embeddings em formato JSON
        embeddings = response.json()['embedding']
        return np.array(embeddings)
    else:
        raise Exception(f"Erro ao gerar embeddings: {response.status_code}, {response.text}")

# Gerando os embeddings
embeddings = get_embeddings(text)

# Exibindo informações sobre os embeddings
print(f"Embeddings gerados para o texto: '{text}'")
print(f"Tamanho do vetor de embeddings: {len(embeddings)}")
print(f"Primeiros 5 valores do vetor: {embeddings[:5]}")

