from services.datasetService import dataset_completo

def prints():
    df = dataset_completo()
    print(df)

if __name__ == "__main__":
    prints()
