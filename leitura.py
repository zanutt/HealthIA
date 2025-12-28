from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y

def print_dataset():
    df = dataset_completo()
    return df

def print_vetorizacao():
    x_tfidf = vetorizacao()
    print(x_tfidf)

def print_encode_Y():
    y_encoded = encode_Y()
    print(y_encoded)

if __name__ == "__main__":
    #print_vetorizacao()
    print_encode_Y()