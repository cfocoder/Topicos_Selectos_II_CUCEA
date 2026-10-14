# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp==2.3.0",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Actividad 1 · Entrenamiento paso a paso de una neurona artificial

    **Nombre:** Hector Gabriel Sanchez Perez

    **Programa:** Maestría en Ciencia de Datos

    **Materia:** Tópicos Selectos de Bases de Datos

    **Fecha:** 11 de octubre de 2026
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actividad 1 · Entrenamiento paso a paso de una neurona artificial

    **Deep Learning para Computer Vision**

    **Nombre:**

    ## Objetivo
    Implementar y explicar el entrenamiento de una neurona para regresión con
    NumPy, desde una predicción inicial hasta la actualización de sus parámetros.
    Al finalizar, deberás relacionar las ecuaciones, las dimensiones de los
    arreglos y el comportamiento de la pérdida durante el aprendizaje.

    Usaremos una activación identidad: este modelo corresponde a una regresión
    lineal. En clase lo abordamos como una primera aproximación al perceptrón.

    ## Instrucciones y entrega
    - Completa las celdas con `TODO` y las respuestas marcadas con **Tu respuesta**.
    - Implementa el modelo con NumPy. No utilices modelos de scikit-learn,
      TensorFlow, PyTorch ni herramientas de diferenciación automática.
    - En cada paso incluye explicación, ecuación, código e interpretación.
    - Conserva la convención de ejemplos en las **columnas** de `X`.
    - Entrega este archivo `.py` de marimo con tus respuestas, código y gráficas.
      Debe ejecutarse completo sin errores y permitir reproducir los resultados.
    - Los bloques incompletos devuelven `None` para que puedas abrir la plantilla.
      Sustitúyelos por tu implementación; no dejes resultados escritos a mano
      como reemplazo del código.

    Esta actividad estudia la mecánica del entrenamiento. Los datos de trabajo
    se usan para entrenar; no se evaluará generalización a datos nuevos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problema y datos

    Predeciremos un **puntaje sintético de rendimiento** a partir de dos
    características numéricas ya preparadas. Estos datos son didácticos y no
    representan mediciones reales. No requieren limpieza ni estandarización.

    1. Identifica las entradas y la salida del modelo.
    2. ¿Por qué este problema es de regresión?
    3. ¿Qué significa un ejemplo de entrenamiento en esta tabla?

    **Respuesta:** Consulta la celda de respuesta situada después de los datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hacen estos datos?** Esta celda crea `X_datos` con las dos características de cada ejemplo y `y_datos` con el puntaje que queremos predecir. Cada columna de `X_datos` corresponde a un ejemplo.
    """)
    return


@app.cell
def _(np):
    X_datos = np.array([
        [-1.0, -0.5, 0.0, 0.5, 1.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, 0.5, -0.5, -1.0, 1.0, 1.0],
    ], dtype=float)
    y_datos = np.array([
        [-0.7, -0.9, 1.3, 1.0, 2.9, 0.6, -0.4, 1.2]
    ], dtype=float)
    return X_datos, y_datos


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué muestra esta tabla?** Organiza los arreglos en filas legibles: cada fila reúne las dos características de un ejemplo y su puntaje objetivo.
    """)
    return


@app.cell
def _(X_datos, mo, y_datos):
    mo.ui.table([
        {"Ejemplo": i + 1, "Característica 1": float(X_datos[0, i]),
         "Característica 2": float(X_datos[1, i]), "Puntaje": float(y_datos[0, i])}
        for i in range(X_datos.shape[1])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Respuestas — Punto 1

    **1.1. Entradas y salida:** Las entradas son las dos características numéricas de cada ejemplo, almacenadas por columnas en `X_datos`. La salida objetivo es el puntaje sintético de rendimiento correspondiente, almacenado en `y_datos`.

    **1.2. ¿Por qué es regresión?:** Porque el modelo predice un puntaje numérico continuo, no una categoría o etiqueta. El objetivo es aproximar ese valor a partir de las características.

    **1.3. Ejemplo de entrenamiento:** Cada ejemplo es una observación individual: un par de valores de características asociado con su puntaje. En los arreglos, cada columna de `X_datos` corresponde a una observación y el valor de `y_datos` en la misma posición es su respuesta objetivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dimensiones e inicialización

    Usaremos $m$ ejemplos, $n_x$ características y una salida:

    | Arreglo | Forma |
    |---|---|
    | $X$ | $(n_x,m)$ |
    | $y$ | $(1,m)$ |
    | $W$ | $(1,n_x)$ |
    | $b$ | $(1,1)$ |
    | $\hat y$ | $(1,m)$ |

    Completa `layer_sizes` e `initialize_parameters`. Inicializa pesos pequeños
    con `np.random.default_rng(seed)` y el sesgo en cero. Para esta neurona,
    inicializar en cero también sería posible; usaremos una semilla para
    comparar experimentos de forma reproducible.

    **Tu respuesta:** indica las dimensiones concretas y explica por qué
    el producto $WX$ y la suma del sesgo son válidos. ¿Qué controla la semilla?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hacen estas funciones?**

    * `layer_sizes` cuenta las características y salidas mirando las formas de los arreglos.
    * `initialize_parameters` crea pesos pequeños reproducibles con la semilla indicada y un sesgo inicial en cero.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **¿Qué muestra esta tabla?** Organiza los arreglos en filas legibles: cada fila reúne las dos características de un ejemplo y su puntaje objetivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Respuesta y explicación — Punto 2

    La función `layer_sizes` obtiene `n_x` del número de filas de `X` (características) y `n_y` del número de filas de `y` (salidas). En estos datos, `X_datos` tiene forma $(2, 8)$ y `y_datos` forma $(1, 8)$; por tanto, $n_x=2$, $n_y=1$ y $m=8$.

    `initialize_parameters` usa `np.random.default_rng(seed)` para generar `W` con forma $(n_y,n_x)=(1,2)$ y valores pequeños (escala 0.01). Inicializa `b` en cero con forma $(n_y,1)=(1,1)`. Con esas dimensiones, $WX$ es $(1,2)(2,8)=(1,8)$; el sesgo $(1,1)$ se difunde a través de las 8 columnas para producir una salida por ejemplo. La semilla fija la secuencia aleatoria y permite reproducir y comparar la inicialización entre ejecuciones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Propagación hacia adelante

    $$Z=WX+b,\qquad \hat y=Z$$

    Implementa `forward_propagation`. Recibe los datos y un diccionario de
    parámetros; devuelve un arreglo con todas las predicciones.

    **Tu respuesta:** ¿qué aportan los pesos y el sesgo? ¿Por qué usamos
    la función identidad para la salida de este ejercicio?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hace esta función?** Calcula una predicción para cada ejemplo: combina sus características con los pesos y suma el sesgo. Devuelve todas las predicciones juntas.
    """)
    return


@app.function
def forward_propagation(X, parameters):
    """Calcula predicciones lineales para ejemplos organizados en columnas."""
    W, b = parameters["W"], parameters["b"]
    return W @ X + b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Respuesta y explicación — Punto 3

    La función toma `W` y `b` del diccionario de parámetros y calcula $Z=WX+b$. Como `W` tiene forma $(1,2)$ y `X_datos` forma $(2,8)$, el producto tiene forma $(1,8)$; el sesgo de forma $(1,1)$ se suma a cada una de las ocho predicciones mediante difusión de NumPy. La función devuelve $Z$ como `y_hat`.

    Los **pesos** determinan cuánto contribuye cada característica a la predicción; el **sesgo** desplaza el valor predicho independientemente de las características. Usamos la **función identidad** ($\hat y=Z$) porque el objetivo es predecir un puntaje numérico continuo mediante regresión lineal. Una activación como la sigmoide restringiría la salida y no corresponde a este objetivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dimensiones e inicialización

    Usaremos $m$ ejemplos, $n_x$ características y una salida:

    | Arreglo | Forma |
    |---|---|
    | $X$ | $(n_x,m)$ |
    | $y$ | $(1,m)$ |
    | $W$ | $(1,n_x)$ |
    | $b$ | $(1,1)$ |
    | $\hat y$ | $(1,m)$ |

    Completa `layer_sizes` e `initialize_parameters`. Inicializa pesos pequeños
    con `np.random.default_rng(seed)` y el sesgo en cero. Para esta neurona,
    inicializar en cero también sería posible; usaremos una semilla para
    comparar experimentos de forma reproducible.

    **Tu respuesta:** indica las dimensiones concretas y explica por qué
    el producto $WX$ y la suma del sesgo son válidos. ¿Qué controla la semilla?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hace esta función?** Compara las predicciones con los valores reales, eleva las diferencias al cuadrado y calcula la pérdida promedio. Un valor menor indica menos error en estos datos.
    """)
    return


@app.cell
def compute_cost(np):
    def layer_sizes(X, y):
        """Devuelve el número de características y salidas de los arreglos."""
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X e y deben ser arreglos bidimensionales")
        if X.shape[1] != y.shape[1]:
            raise ValueError("X e y deben contener el mismo número de ejemplos")
        return X.shape[0], y.shape[0]


    def initialize_parameters(n_x, n_y, seed=42):
        """Inicializa pesos pequeños reproducibles y sesgo cero."""
        rng = np.random.default_rng(seed)
        return {"W": rng.standard_normal((n_y, n_x)) * 0.01,
                "b": np.zeros((n_y, 1))}


    def compute_cost(y_hat, y):
        """Calcula el error cuadrático medio con el factor 1/2."""
        if y_hat.shape != y.shape:
            raise ValueError("y_hat e y deben tener la misma forma")
        m = y.shape[1]
        if m == 0:
            raise ValueError("Se requiere al menos un ejemplo")
        return float(np.sum((y_hat - y) ** 2) / (2 * m))


    def back_propagation(X, y, parameters, y_hat):
        """Calcula gradientes promedio para pesos y sesgo."""
        m = X.shape[1]
        if m == 0 or y_hat.shape != y.shape or y.shape[1] != m:
            raise ValueError("Dimensiones incompatibles o conjunto vacío")
        dZ = y_hat - y
        return {"dW": (dZ @ X.T) / m,
                "db": np.sum(dZ, axis=1, keepdims=True) / m}


    def update_parameters(parameters, gradients, learning_rate=0.1):
        """Devuelve parámetros nuevos sin modificar los arreglos originales."""
        return {"W": parameters["W"] - learning_rate * gradients["dW"],
                "b": parameters["b"] - learning_rate * gradients["db"]}

    return (
        back_propagation,
        compute_cost,
        initialize_parameters,
        layer_sizes,
        update_parameters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Respuesta y explicación — Punto 4

    La función calcula el error entre cada predicción y su valor objetivo, eleva cada error al cuadrado, suma los errores y divide entre $2m$. Así devuelve un único escalar que mide la pérdida promedio del conjunto, con el factor $1/2$ indicado en la fórmula.

    Elevamos al cuadrado para que los errores positivos y negativos no se cancelen y para penalizar más los errores de mayor magnitud. El factor $1/2$ simplifica la derivada: al derivar el cuadrado aparece un factor 2 que se cancela con ese $1/2$. Una pérdida cero **no garantiza** buen desempeño con datos nuevos; solo indica predicción perfecta en los datos evaluados. Aquí la actividad se enfoca en mecánica de entrenamiento y no evalúa generalización.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Gradientes y actualización

    Con pesos organizados como **fila**, las ecuaciones son:

    \[
    dZ=\hat y-y,\qquad dW=\frac{1}{m}dZ X^T,
    \qquad db=\frac{1}{m}\sum_{i=1}^{m}dZ^{(i)}\]

    \[W_{\mathrm{nuevo}}=W-\alpha dW,\qquad
    b_{\mathrm{nuevo}}=b-\alpha db\]

    Implementa las dos funciones. Conserva las formas de `dW` y `db`;
    `np.sum(..., axis=1, keepdims=True)` puede ayudarte con el sesgo.
    La actualización debe devolver un diccionario nuevo sin modificar
    los arreglos originales en el lugar.

    **Tu respuesta:** muestra la derivación de $\partial J/\partial W_j$
    y $\partial J/\partial b$. Explica por qué restamos el gradiente y
    qué representa la tasa de aprendizaje $\alpha$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hacen estas funciones?**
    * `back_propagation` mide cómo contribuye cada peso y el sesgo al error.
    * `update_parameters` usa esos gradientes y la tasa de aprendizaje para ajustar los parámetros sin modificar los originales.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Respuesta y explicación — Punto 2

    La función `layer_sizes` obtiene `n_x` del número de filas de `X` (características) y `n_y` del número de filas de `y` (salidas). En estos datos, `X_datos` tiene forma $(2, 8)$ y `y_datos` forma $(1, 8)$; por tanto, $n_x=2$, $n_y=1$ y $m=8$.

    `initialize_parameters` usa `np.random.default_rng(seed)` para generar `W` con forma $(n_y,n_x)=(1,2)$ y valores pequeños (escala 0.01). Inicializa `b` en cero con forma $(n_y,1)=(1,1)`. Con esas dimensiones, $WX$ es $(1,2)(2,8)=(1,8)$; el sesgo $(1,1)$ se difunde a través de las 8 columnas para producir una salida por ejemplo. La semilla fija la secuencia aleatoria y permite reproducir y comparar la inicialización entre ejecuciones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Respuesta y explicación — Punto 5

    La propagación hacia atrás calcula primero el error de cada ejemplo: `dZ = y_hat - y`. Para obtener el gradiente de cada peso, se multiplica ese error por el valor de la característica correspondiente y se promedian los resultados. En conjunto, el código lo expresa como `dW = (dZ @ X.T) / m`.

    El sesgo afecta por igual a todas las predicciones, así que su gradiente es el promedio de los errores: `db = sum(dZ, axis=1, keepdims=True) / m`.

    `back_propagation` devuelve esos dos gradientes. Luego `update_parameters` resta a cada parámetro su gradiente multiplicado por la tasa de aprendizaje: `W_nuevo = W - alpha * dW` y `b_nuevo = b - alpha * db`. Restamos porque el gradiente indica hacia dónde aumenta la pérdida; avanzar en sentido contrario busca reducirla. `alpha` controla el tamaño del paso: si es muy grande puede sobrepasar el mínimo o divergir, y si es muy pequeña el aprendizaje puede ser lento.

    En la prueba manual, el factor `1/2` de la pérdida se cancela con el `2` que aparece al derivar el error al cuadrado; por eso las fórmulas de los gradientes no llevan un factor adicional de 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Propagación hacia adelante

    $$Z=WX+b,\qquad \hat y=Z$$

    Implementa `forward_propagation`. Recibe los datos y un diccionario de
    parámetros; devuelve un arreglo con todas las predicciones.

    **Tu respuesta:** ¿qué aportan los pesos y el sesgo? ¿Por qué usamos
    la función identidad para la salida de este ejercicio?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué comprueba este cálculo?** Usa un conjunto pequeño de dos ejemplos para aplicar una sola actualización. Muestra predicciones, pérdida y gradientes antes y después, de modo que podamos compararlos con el desarrollo manual.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **¿Qué hace esta función?** Calcula una predicción para cada ejemplo: combina sus características con los pesos y suma el sesgo. Devuelve todas las predicciones juntas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Desarrollo manual y respuesta — Punto 6

    Con los dos ejemplos, las predicciones iniciales son `y_hat = [[0, 0]]`, porque ambos pesos y el sesgo empiezan en cero. Los errores son `dZ = y_hat - y = [[-2, -3]]`.

    La pérdida inicial es `( (-2)^2 + (-3)^2 ) / (2 × 2) = 13/4 = 3.25`.

    Para los pesos, multiplicamos los errores por los valores de cada característica y promediamos:

    - Primer peso: `((-2 × 1) + (-3 × 2)) / 2 = -4`.
    - Segundo peso: `((-2 × 0) + (-3 × 1)) / 2 = -1.5`.
    - Sesgo: `(-2 + -3) / 2 = -2.5`.

    Con `alpha = 0.1`, la actualización es `W = [0, 0] - 0.1 × [-4, -1.5] = [0.4, 0.15]` y `b = 0 - 0.1 × (-2.5) = 0.25`. Las nuevas predicciones son `[[0.65, 1.20]]`. La pérdida posterior es `( (0.65 - 2)^2 + (1.20 - 3)^2 ) / 4 = 1.265625`.

    NumPy debe producir esos mismos resultados (salvo posibles diferencias de redondeo). La pérdida disminuyó de `3.25` a `1.265625`; las predicciones aumentaron desde cero hacia los valores objetivo `2` y `3`, aunque todavía no coinciden exactamente después de una sola actualización.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Ciclo de entrenamiento

    Integra tus funciones en `train_model`:

    1. Obtén las dimensiones usando los argumentos `X` e `y`.
    2. Inicializa los parámetros una sola vez.
    3. Guarda la pérdida inicial.
    4. En cada iteración: predice, calcula gradientes, actualiza parámetros
       y guarda la pérdida **después** de actualizar.
    5. Devuelve los parámetros finales y el historial de pérdidas.

    El historial debe tener `iterations + 1` valores: el índice cero es
    la pérdida antes de entrenar. Usa únicamente los datos recibidos por
    la función, para que también funcione con el conjunto del cálculo manual.

    Entrena con 200 iteraciones, tasa 0.1 y semilla 42. Muestra los parámetros,
    las predicciones finales, las pérdidas inicial y final y una gráfica
    de pérdida contra número de actualizaciones.

    **Tu respuesta:** describe qué se repite y qué se conserva en cada iteración.
    ¿Una iteración de este entrenamiento utiliza todos los ejemplos?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hace esta función?** Repite el ciclo de predicción, cálculo de gradientes y ajuste de parámetros. Guarda la pérdida inicial y la pérdida después de cada actualización, y devuelve los parámetros finales.
    """)
    return


@app.cell
def train_model(
    back_propagation,
    compute_cost,
    initialize_parameters,
    layer_sizes,
    update_parameters,
):
    def train_model(X, y, iterations=200, learning_rate=0.1, seed=42):
        """Entrena por descenso de gradiente batch y conserva la pérdida."""
        if iterations < 0:
            raise ValueError("iterations debe ser no negativo")
        n_x, n_y = layer_sizes(X, y)
        parameters = initialize_parameters(n_x, n_y, seed=seed)
        history = [compute_cost(forward_propagation(X, parameters), y)]
        for _ in range(iterations):
            y_hat = forward_propagation(X, parameters)
            gradients = back_propagation(X, y, parameters, y_hat)
            parameters = update_parameters(parameters, gradients, learning_rate)
            history.append(compute_cost(forward_propagation(X, parameters), y))
        return parameters, history

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hace este experimento?** Entrena con el dataset principal durante 200 actualizaciones y muestra los parámetros, las predicciones y las pérdidas. La gráfica permite ver cómo cambia el error durante el aprendizaje.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Función de pérdida

    $$J(W,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2$$

    Implementa `compute_cost`: debe devolver un escalar. Utiliza `np.sum`;
    si aparecen valores no finitos, investiga su causa.

    **Tu respuesta:** ¿por qué elevamos el error al cuadrado? ¿Qué efecto
    tiene el factor $1/2$ en la derivada? ¿Pérdida cero implica siempre
    que el modelo funcionará bien con datos nuevos?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Interpretación de la gráfica:** La pérdida desciende conforme avanzan las actualizaciones y se aplana cerca del final. Esto indica que el modelo va reduciendo el error y que, con esta tasa de aprendizaje, los cambios se hacen cada vez menores. La gráfica muestra el ajuste al dataset de entrenamiento; no evalúa datos nuevos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Respuesta y explicación — Punto 7

    `train_model` obtiene las dimensiones de los datos que recibe, inicializa `W` y `b` una sola vez y guarda la pérdida antes de entrenar. En cada iteración repite cuatro pasos: calcula predicciones, obtiene gradientes, actualiza los parámetros y vuelve a calcular la pérdida con los parámetros nuevos. Así conserva los parámetros actualizados y el historial de pérdidas; el historial tiene 201 valores: el inicial y uno después de cada una de las 200 actualizaciones.

    Cada iteración **sí utiliza todos los ejemplos**: las predicciones se calculan sobre todas las columnas de `X`, y los gradientes promedian los errores de los `m` ejemplos. Por eso es descenso por gradiente batch, no una actualización basada en un solo ejemplo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué hace esta función?** Compara las predicciones con los valores reales, eleva las diferencias al cuadrado y calcula la pérdida promedio. Un valor menor indica menos error en estos datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **¿Qué compara este experimento?** Repite el entrenamiento con tres tasas de aprendizaje, reiniciando los parámetros y usando los mismos datos y semilla. La tabla resume los resultados y la gráfica compara las curvas de pérdida.
    """)
    return


@app.cell
def _(X_datos, mo, np, plt, train_model, y_datos):
    learning_rates = [0.01, 0.1, 3.0]
    experiment_results = {}
    experiment_rows = []

    for lr in learning_rates:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            params_lr, history_lr = train_model(
                X_datos, y_datos,
                iterations=200,
                learning_rate=lr,
                seed=42,
            )
        history_lr = np.asarray(history_lr, dtype=float)
        experiment_results[lr] = {"parameters": params_lr, "history": history_lr}
        finite_count = int(np.isfinite(history_lr).sum())
        all_finite = finite_count == len(history_lr)
        if not all_finite or history_lr[-1] > history_lr[0] * 1000:
            behavior = "Divergencia / crecimiento explosivo"
        elif abs(history_lr[-1] - history_lr[-10]) / max(abs(history_lr[-10]), 1e-15) < 1e-5:
            behavior = "Convergencia (pérdida casi estable al final)"
        else:
            behavior = "Descenso estable, aún lento"
        finite_note = "Sí (201/201 valores finitos)" if all_finite else f"No ({finite_count}/{len(history_lr)} valores finitos)"
        experiment_rows.append({
            "Tasa de aprendizaje": lr,
            "Pérdida inicial": float(history_lr[0]),
            "Pérdida final": float(history_lr[-1]),
            "Comportamiento": behavior,
            "Valores finitos": finite_note,
        })

    fig_experimento, ax_experimento = plt.subplots(figsize=(8, 5))
    for lr in learning_rates:
        h = experiment_results[lr]["history"]
        finite_positive = np.isfinite(h) & (h > 0)
        ax_experimento.semilogy(np.arange(len(h))[finite_positive], h[finite_positive], label=f"α = {lr}")
    ax_experimento.set_xlabel("Número de actualizaciones")
    ax_experimento.set_ylabel("Pérdida (escala logarítmica)")
    ax_experimento.set_title("Efecto de la tasa de aprendizaje")
    ax_experimento.legend()
    ax_experimento.grid(True, which="both", alpha=0.3)

    mo.vstack([mo.ui.table(experiment_rows), fig_experimento])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Interpretación de la gráfica:** En escala logarítmica se comparan las tres tasas: `0.01` reduce la pérdida de forma estable pero lenta; `0.1` la reduce más rápido y se estabiliza; `3.0` empieza bajando, pero luego crece de forma explosiva, señal de divergencia. Una tasa mayor puede dar pasos demasiado grandes y alejar el modelo de una solución estable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Análisis y respuesta — Punto 8

    Los tres experimentos usan `X_datos` y `y_datos`, 200 actualizaciones y semilla 42; `train_model` vuelve a inicializar los parámetros en cada llamada. La tasa `0.1` es la mejor de estas opciones para este experimento: reduce la pérdida más y se estabiliza cerca del final. La tasa `0.01` también desciende de forma estable, pero aprende más lentamente. Con `3.0`, la pérdida primero baja un poco y después crece de forma explosiva; al terminar alcanza aproximadamente `1.22 × 10^132`. Los 201 valores de cada historial son finitos, aunque la pérdida con tasa 3.0 evidencia divergencia numérica/optimización inestable.

    Una tasa mayor no garantiza aprender mejor: los pasos pueden sobrepasar repetidamente la región de menor pérdida y hacer que los parámetros y el error crezcan. Con 200 actualizaciones solo podemos comparar estas tasas para este dataset, inicialización y configuración; no basta para afirmar que `0.1` sea óptima en general ni para concluir cómo generalizaría el modelo a datos nuevos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Conclusiones y conexión con Deep Learning

    Responde con referencia a tus resultados:

    1. ¿Qué significa que la neurona haya aprendido? Relaciona parámetros,
       predicciones y pérdida.
    2. ¿Qué diferencia existe entre calcular gradientes y actualizar parámetros?
    3. ¿Cómo comprobarías si una implementación entrega resultados correctos,
       además de observar que la pérdida disminuye?
    4. Si usamos sigmoide y entropía cruzada para clasificación binaria,
       ¿qué partes del proceso cambian y cuáles conservan su estructura?

    ### Conclusiones

    **1. Aprendizaje:** La neurona aprende cuando los ajustes de `W` y `b` hacen que sus predicciones se acerquen a los puntajes objetivo y disminuya la pérdida. En este ejercicio, con tasa `0.1`, la pérdida del dataset principal bajó de `0.891738` a `0.022355` en 200 actualizaciones. Esto muestra ajuste a estos datos, no garantiza por sí solo desempeño con datos nuevos.

    **2. Gradientes y actualización:** Calcular gradientes estima cómo cambia la pérdida al modificar cada parámetro; no cambia los parámetros. La actualización usa esos gradientes y la tasa de aprendizaje para producir nuevos `W` y `b`. En la comprobación manual, se calcularon primero `dW` y `db`, y después se aplicó la regla de resta.

    **3. Verificación:** Además de revisar que la pérdida baje, comprobaría dimensiones y tipos, probaría casos pequeños con resultados calculados a mano, compararía gradientes con aproximaciones numéricas por diferencias finitas y probaría casos límite (por ejemplo, predicción perfecta y entradas no finitas). También verificaría reproducibilidad con la misma semilla y que las funciones no modifiquen parámetros originales por accidente. La iteración manual de este notebook sirve como una de esas pruebas.

    **4. Clasificación binaria:** Cambiarían la activación de salida —sigmoide para obtener una probabilidad entre 0 y 1— y la pérdida, que sería entropía cruzada binaria en lugar del error cuadrático. También cambiaría el gradiente de salida correspondiente; con sigmoide y entropía cruzada suele simplificarse a `dZ = y_hat - y`. Se conservan la inicialización de parámetros, la propagación hacia adelante y hacia atrás, el cálculo de gradientes, la actualización con descenso por gradiente y el ciclo de entrenamiento, ajustados a las dimensiones y al objetivo de clasificación.

    ## Rúbrica

    | Criterio | Valor | Evidencia esperada |
    |---|---:|---|
    | Implementación correcta | 40 % | Dimensiones coherentes, funciones propias, actualización y ciclo completos; coincidencia con la iteración manual. |
    | Explicación matemática | 30 % | Ecuaciones, derivación de gradientes y explicación del papel de cada paso. |
    | Experimentación e interpretación | 20 % | Comparación controlada de tasas, gráficas, tabla y conclusiones sustentadas. |
    | Claridad y reproducibilidad | 10 % | Notebook ordenado, semilla fija, respuestas completas y ejecución sin errores. |

    Revisa tus resultados antes de entregar.
    """)
    return


if __name__ == "__main__":
    app.run()
