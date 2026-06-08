def leer_csv(archivo):
    """Lee un CSV y devuelve los atributos de entrada (X) y la clase (y)."""
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    # La primera fila contiene los nombres de las columnas
    encabezado = lineas[0].strip().split(',')

    # Columnas que NO se usan como entrada (fecha, ubicación y la propia clase)
    excluir = ['day', 'month', 'year', 'region', 'classes']
    # Índices de las columnas que sí son atributos de entrada
    indices_entrada = [i for i, col in enumerate(encabezado)
                       if col.strip().lower() not in excluir]
    # Índice de la columna que contiene la clase a predecir
    indice_clase = next(i for i, col in enumerate(encabezado) if 'Classes' in col)

    X = []  # lista de vectores de atributos
    y = []  # lista de clases (0 o 1)
    for linea in lineas[1:]:  # se recorren todas las filas menos el encabezado
        cols = linea.strip().split(',')
        try:
            # Se convierten los atributos a número decimal
            entrada = [float(cols[i].strip()) for i in indices_entrada]
        except ValueError:
            # Si alguna fila tiene datos no numéricos, se descarta
            continue
        # Se codifica la clase: 1 = "fire", 0 = cualquier otro valor
        clase = 1 if cols[indice_clase].strip().lower() == 'fire' else 0
        X.append(entrada)
        y.append(clase)

    return X, y


def calcular_minmax(X):
    """Calcula el valor mínimo y máximo de cada atributo (columna)."""
    n_atributos = len(X[0])
    minimos = [min(fila[i] for fila in X) for i in range(n_atributos)]
    maximos = [max(fila[i] for fila in X) for i in range(n_atributos)]
    return minimos, maximos


def normalizar(X, minimos, maximos):
    """Escala cada atributo al rango [0, 1] usando normalización Min-Max.

    Se usan los mínimos/máximos del set de entrenamiento para no filtrar
    información del set de prueba.
    """
    X_norm = []
    for fila in X:
        fila_norm = []
        for i in range(len(fila)):
            rango = maximos[i] - minimos[i]
            if rango == 0:
                # Si el atributo es constante, se evita la división por cero
                fila_norm.append(0.0)
            else:
                fila_norm.append((fila[i] - minimos[i]) / rango)
        X_norm.append(fila_norm)
    return X_norm


def predecir(pesos, bias, entrada):
    """Calcula la salida del perceptrón para una entrada.

    Aplica la suma ponderada (pesos·entrada + bias) y una función de
    activación escalón: devuelve 1 si la suma es >= 0, y 0 en caso contrario.
    """
    suma = bias
    for i in range(len(pesos)):
        suma += pesos[i] * entrada[i]
    return 1 if suma >= 0 else 0


def entrenar(X_train, y_train, tasa=0.1, epocas=100):
    """Entrena el perceptrón con la regla de aprendizaje clásica.

    tasa  : tasa de aprendizaje (cuánto se ajustan los pesos por error).
    epocas: número máximo de pasadas completas sobre los datos.
    """
    n_atributos = len(X_train[0])
    pesos = [0.0] * n_atributos  # pesos iniciales en cero
    bias = 0.0                   # sesgo (umbral) inicial en cero

    for epoca in range(epocas):
        errores = 0
        # Se recorre cada ejemplo de entrenamiento
        for entrada, clase_real in zip(X_train, y_train):
            prediccion = predecir(pesos, bias, entrada)
            error = clase_real - prediccion  # 0 si acierta, +1 o -1 si falla
            if error != 0:
                errores += 1
                # Regla de actualización: se ajustan pesos y bias según el error
                for i in range(len(pesos)):
                    pesos[i] += tasa * error * entrada[i]
                bias += tasa * error

        print(f"Época {epoca + 1}: {errores} errores")

        # Si no hubo errores, los datos son linealmente separables y se detiene
        if errores == 0:
            print(f"Convergió en época {epoca + 1}")
            break

    return pesos, bias


def evaluar(pesos, bias, X_test, y_test):
    """Evalúa el modelo sobre el set de prueba y muestra la matriz de confusión."""
    # matriz[clase_real][prediccion] -> cantidad de casos
    matriz = [[0, 0], [0, 0]]

    for entrada, clase_real in zip(X_test, y_test):
        prediccion = predecir(pesos, bias, entrada)
        matriz[clase_real][prediccion] += 1

    # Se extraen las cuatro celdas de la matriz de confusión
    vn = matriz[0][0]  # verdaderos negativos: real not fire, predijo not fire
    fp = matriz[0][1]  # falsos positivos:     real not fire, predijo fire
    fn = matriz[1][0]  # falsos negativos:     real fire,     predijo not fire
    vp = matriz[1][1]  # verdaderos positivos: real fire,     predijo fire

    total = len(y_test)
    # Accuracy = aciertos / total (en porcentaje)
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
# Este bloque solo se ejecuta al correr el archivo directamente
# (python perceptron.py), no cuando se importan sus funciones desde otro módulo.
if __name__ == "__main__":
    # 1) Se cargan los datos de entrenamiento y prueba
    # X_train: lista de filas, cada fila es una lista de atributos (float)
    #   X_train = [[29.0, 57.0, 18.0, 0.0],   <- ejemplo 1
    #              [26.0, 82.0, 22.0, 1.3],   <- ejemplo 2
    #              ...]
    # y_train: lista de clases (1 = fire, 0 = not fire), una por fila
    #   y_train = [0, 1, 1, 0, ...]
    X_train, y_train = leer_csv('train.csv')
    X_test, y_test   = leer_csv('test.csv')

    # 2) Se normalizan los atributos al rango [0, 1] usando los min/max del train
    minimos, maximos = calcular_minmax(X_train)
    X_train_norm = normalizar(X_train, minimos, maximos)
    X_test_norm  = normalizar(X_test,  minimos, maximos)

    # 3) Se entrena el perceptrón
    print("=== ENTRENAMIENTO ===")
    pesos, bias = entrenar(X_train_norm, y_train, tasa=0.1, epocas=100)

    # 4) Se evalúa el modelo sobre datos no vistos
    evaluar(pesos, bias, X_test_norm, y_test)
