from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y
from services.treinamentoService import acuracia_modelo

def print_dataset():
    df = dataset_completo()
    return df

def print_vetorizacao():
    x_tfidf = vetorizacao()
    print(x_tfidf)

def print_encode_Y():
    y_encoded = encode_Y()
    print(y_encoded)

def acuracia_modelo_print():
    acuracia = acuracia_modelo()
    print(f"Acuracia do modelo: {acuracia:.2f}%")

if __name__ == "__main__":
    #print_vetorizacao()
    # print_encode_Y()
    acuracia_modelo_print()