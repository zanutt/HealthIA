import pandas as pd

# 1️⃣ Ler o CSV
df = pd.read_csv("dataset_texto_livre_limpo.csv")

# Ver as primeiras linhas e colunas disponíveis (diagnóstico rápido)
print(df.head())
print("\nColunas no CSV:", list(df.columns))

# detectar automaticamente as colunas de texto e rótulo quando possível
def _find_column(df, candidates):
    """Retorna o primeiro nome de coluna presente em df a partir da lista candidates (case-insensitive).
    Se nenhum for encontrado, retorna None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# listas de nomes possíveis para texto e rótulo
text_candidates = ["texto", "sintomas_texto", "sintomas", "sintoma", "text", "texto_livre", "sintomas_raw"]
label_candidates = ["rotulo", "label", "diagnostico", "diagnosis", "rotulo_orig", "diagnostico_limpo", "diagnostico_clean"]

text_col = _find_column(df, text_candidates)
label_col = _find_column(df, label_candidates)

if text_col is None or label_col is None:
    # tenta correspondência parcial por substring
    if text_col is None:
        for c in df.columns:
            if "texto" in c.lower() or "sintom" in c.lower() or "text" in c.lower():
                text_col = c
                break
    if label_col is None:
        for c in df.columns:
            if "rotul" in c.lower() or "diagn" in c.lower() or "label" in c.lower():
                label_col = c
                break

if text_col is None or label_col is None:
    raise KeyError(
        "Arquivo CSV não contém colunas reconhecíveis para texto/rotulo. Colunas disponíveis: "
        + ", ".join(df.columns)
        + ".\nProcure por um nome como 'texto' ou 'sintomas_texto' para os textos e 'rotulo' ou 'diagnostico' para os rótulos."
    )

print(f"Usando coluna de texto: '{text_col}' e coluna de rótulo: '{label_col}'")

# 2️⃣ Normalizar os textos
def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower().strip()  # deixa minúsculo e remove espaços
    # remove pontuações desnecessárias
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'"]:
        texto = texto.replace(ch, "")
    # remove espaços duplos
    texto = " ".join(texto.split())
    return texto

# aplica a limpeza nas colunas detectadas
df["texto_limpo"] = df[text_col].apply(limpar_texto)
df["rotulo_limpo"] = df[label_col].apply(limpar_texto)

# 5️⃣ Salvar resultado em CSV — inclui as colunas originais detectadas e as colunas limpas
output_csv = "dataset_texto_livre_limpo_processado.csv"
cols_to_save = [text_col, label_col, "texto_limpo", "rotulo_limpo"]
try:
    df.to_csv(output_csv, index=False, columns=cols_to_save)
    print(f"\nArquivo processado salvo em: {output_csv}")
except Exception as e:
    print(f"Erro ao salvar arquivo '{output_csv}': {e}")

# 3️⃣ Agrupar os exemplos por doença
grupos = df.groupby("rotulo_limpo")["texto_limpo"].apply(list)

# 4️⃣ Gerar formato Python igual ao do vídeo
print("\ndados = [")
for rotulo, exemplos in grupos.items():
    print(f'    # {rotulo.capitalize()} ({len(exemplos)} exemplos)')
    for texto in exemplos:
        print(f'    ("{texto}", "{rotulo}"),')
    print()
print("]")
