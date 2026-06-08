
import random

def split_estratificado(archivo, porcentaje_test=0.2, semilla=42):
    random.seed(semilla)

    with open(archivo, 'r') as f:
        lineas = f.readlines()

    # Filtrar encabezados de región y encabezados de columnas
    encabezado = None
    datos = []
    for linea in lineas:
        limpia = linea.strip()
        if limpia == '' or 'Region Dataset' in limpia:
            continue
        if limpia.startswith('day'):
            if encabezado is None:
                encabezado = linea  # guardar solo el primero
            continue
        datos.append(linea)

    print(f"Total instancias leídas: {len(datos)}")

    # Separar por clase
    grupos = {}
    for linea in datos:
        clase = linea.strip().split(',')[-1].strip()
        if clase not in grupos:
            grupos[clase] = []
        grupos[clase].append(linea)

    print("Distribución original:")
    for clase, instancias in grupos.items():
        print(f"  {clase}: {len(instancias)}")

    # Mezclar y dividir dentro de cada grupo
    train = []
    test = []
    for clase, instancias in grupos.items():
        random.shuffle(instancias)
        n_test = round(len(instancias) * porcentaje_test)
        test  += instancias[:n_test]
        train += instancias[n_test:]

    # Guardar archivos
    with open('train.csv', 'w') as f:
        f.write(encabezado)
        f.writelines(train)

    with open('test.csv', 'w') as f:
        f.write(encabezado)
        f.writelines(test)

    print(f"\nTrain: {len(train)} instancias")
    print(f"Test:  {len(test)} instancias")
    print("\nDistribución en train:")
    conteo_train = {}
    for linea in train:
        clase = linea.strip().split(',')[-1].strip()
        conteo_train[clase] = conteo_train.get(clase, 0) + 1
    for clase, cantidad in conteo_train.items():
        pct = round(cantidad / len(train) * 100, 1)
        print(f"  {clase}: {cantidad} ({pct}%)")

    print("\nDistribución en test:")
    conteo_test = {}
    for linea in test:
        clase = linea.strip().split(',')[-1].strip()
        conteo_test[clase] = conteo_test.get(clase, 0) + 1
    for clase, cantidad in conteo_test.items():
        pct = round(cantidad / len(test) * 100, 1)
        print(f"  {clase}: {cantidad} ({pct}%)")
    

split_estratificado('Algerian_forest_fires_dataset_UPDATE.csv')

