def leer_csv(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    encabezado = lineas[0].strip().split(',')
    excluir = ['day', 'month', 'year', 'region', 'classes']
    indices_entrada = [i for i, col in enumerate(encabezado)
                       if col.strip().lower() not in excluir]
    indice_clase = next(i for i, col in enumerate(encabezado) if 'Classes' in col)

    X = []
    y = []
    for linea in lineas[1:]:
        cols = linea.strip().split(',')
        try:
            entrada = [float(cols[i].strip()) for i in indices_entrada]
        except ValueError:
            continue
        clase = 1 if cols[indice_clase].strip().lower() == 'fire' else 0
        X.append(entrada)
        y.append(clase)

    return X, y


def calcular_minmax(X):
    n_atributos = len(X[0])
    minimos = [min(fila[i] for fila in X) for i in range(n_atributos)]
    maximos = [max(fila[i] for fila in X) for i in range(n_atributos)]
    return minimos, maximos


def normalizar(X, minimos, maximos):
    X_norm = []
    for fila in X:
        fila_norm = []
        for i in range(len(fila)):
            rango = maximos[i] - minimos[i]
            if rango == 0:
                fila_norm.append(0.0)
            else:
                fila_norm.append((fila[i] - minimos[i]) / rango)
        X_norm.append(fila_norm)
    return X_norm


def predecir(pesos, bias, entrada):
    suma = bias
    for i in range(len(pesos)):
        suma += pesos[i] * entrada[i]
    return 1 if suma >= 0 else 0


def entrenar(X_train, y_train, tasa=0.1, epocas=100):
    n_atributos = len(X_train[0])
    pesos = [0.0] * n_atributos
    bias = 0.0

    for epoca in range(epocas):
        errores = 0
        for entrada, clase_real in zip(X_train, y_train):
            prediccion = predecir(pesos, bias, entrada)
            error = clase_real - prediccion
            if error != 0:
                errores += 1
                for i in range(len(pesos)):
                    pesos[i] += tasa * error * entrada[i]
                bias += tasa * error

        print(f"Época {epoca + 1}: {errores} errores")

        if errores == 0:
            print(f"Convergió en época {epoca + 1}")
            break

    return pesos, bias


def evaluar(pesos, bias, X_test, y_test):
    matriz = [[0, 0], [0, 0]]

    for entrada, clase_real in zip(X_test, y_test):
        prediccion = predecir(pesos, bias, entrada)
        matriz[clase_real][prediccion] += 1

    vn = matriz[0][0]
    fp = matriz[0][1]
    fn = matriz[1][0]
    vp = matriz[1][1]

    total = len(y_test)
    accuracy = (vp + vn) / total * 100

    print("\n=== MATRIZ DE CONFUSIÓN ===")
    print(f"{'':20} {'Pred. not fire':>15} {'Pred. fire':>12}")
    print(f"{'Real not fire':20} {vn:>15} {fp:>12}")
    print(f"{'Real fire':20} {fn:>15} {vp:>12}")
    print(f"\nVerdaderos positivos (VP): {vp}")
    print(f"Verdaderos negativos (VN): {vn}")
    print(f"Falsos positivos     (FP): {fp}")
    print(f"Falsos negativos     (FN): {fn}")
    print(f"\nAccuracy: {accuracy:.1f}%")


# === PROGRAMA PRINCIPAL ===
X_train, y_train = leer_csv('train.csv')
X_test, y_test   = leer_csv('test.csv')

minimos, maximos = calcular_minmax(X_train)
X_train_norm = normalizar(X_train, minimos, maximos)
X_test_norm  = normalizar(X_test,  minimos, maximos)

print("=== ENTRENAMIENTO ===")
pesos, bias = entrenar(X_train_norm, y_train, tasa=0.1, epocas=100)

evaluar(pesos, bias, X_test_norm, y_test)