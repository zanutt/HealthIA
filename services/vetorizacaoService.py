# 1 - Chamar o Framework que tem o TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo

# 2 - Instanciar o modelo de vetorizacao
def vetorizacao():
    # Instanciar o modelo
    tfidf = TfidfVectorizer()

    # Chama o dataset importado
    df = dataset_completo()
    # Selecionar a coluna de sintomas como string
    x = df['sintomas'].astype(str)

    # Treinar o modelo e transformar os dados em vetores
    tfidf.fit(x)
    x_tfidf = tfidf.transform(x)

    # Retornar os dados vetorizados 
    return x_tfidf
    
# 3 - Pegar os dados e vetorizar