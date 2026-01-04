# 1 - Buscar os dados - x,y

# 2 - Separar os dados em treino e teste

# 3 - Treinar o modelo

# 4 - Testar o modelo ver a acuracia


from services.vetorizacaoService import vetorizacao, encode_Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Buscar o modelo XGBoost
from xgboost import XGBClassifier as XGBoost

def buscar_dados():
    print("Carregando dados vetorizados e labels...")
    x = vetorizacao()
    y = encode_Y()
    print("Dados carregados para X e Y.")
    return x, y

def separar_dados():
    print("Separando dados em treino (80%) e teste (20%)...")
    x, y = buscar_dados()
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
    print(
        f"Dados separados: treino={x_treino.shape[0]} amostras, teste={x_teste.shape[0]} amostras."
    )
    return x_treino, x_teste, y_treino, y_teste

def treinar_modelo():
    x_treino, x_teste, y_treino, y_teste = separar_dados()

    print("Treinando o modelo XGBoost...")
    healthIA = XGBoost()
    healthIA.fit(x_treino, y_treino)
    print("Treinamento concluido.")

    return healthIA, x_teste, y_teste

def acuracia_modelo():
    print("Iniciando avaliacao de acuracia...")
    healthIA, x_teste, y_teste = treinar_modelo()

    print("Gerando predicoes no conjunto de teste...")
    y_pred = healthIA.predict(x_teste)

    acuracia = accuracy_score(y_teste, y_pred)
    porcentagem = acuracia * 100

    print(f"Acuracia obtida: {porcentagem:.2f}%")
    return porcentagem