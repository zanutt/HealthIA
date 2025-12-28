
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from services.datasetService import dataset_completo

# Instanciar o modelo de vetorizacao
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
    
def encode_Y():
    # instanciar o label encoder
    label_encoder = LabelEncoder()

    # Chama o dataset importado
    df = dataset_completo()

    # Selecionar a coluna de diagnosticos como string
    y = df['diagnostico'].astype(str)

    # Ajustar e transformar os rótulos em valores numéricos
    y_encoded = label_encoder.fit_transform(y)

    # Retornar os rótulos codificados
    return y_encoded