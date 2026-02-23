"""
«La manera más atómica de entrenar e inferir un GPT en Python puro e independiente.
Este archivo contiene el algoritmo completo.
El resto es mera eficiencia.»

@karpathy
"""

import os       # Verificar si existe el archivo de entrada mediante os.path.exists
import math     # Funciones logarítmicas y potencias mediante math.log, math.exp
import random   # Funciones aleatorias mediante random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Iniciar el programa en un estado reproducible

archivo = "nombres.txt"
# Conjunto de datos de entrada: lista de «documentos» (tal como un conjunto de nombres)
if not os.path.exists(archivo):
    import urllib.request
    nombres_direccion = 'https://raw.githubusercontent.com/mgcamacho/microgpt/prima/nombres.txt'
    urllib.request.urlretrieve(nombres_direccion, archivo)
documentos = [linea.strip() for linea in open(archivo, encoding="latin1").read().strip().split('\n') if linea.strip()] # lista[str] de documentos
random.shuffle(documentos)
print(f"cantidad docs: {len(documentos)}")

# Fichador que traduce cadenas de texto a secuencias de enteros («fichas») y viceversa
caracteres_unicos = sorted(set(''.join(documentos))) # Los distintos caracteres en el conjunto son transformados en fichas 0..n-1
PDS = len(caracteres_unicos) # Identificador de la ficha especial de principio de secuencia (PDS)
long_vocab = len(caracteres_unicos) + 1 # Cantidad total de fichas únicas, +1 para el PDS
print(f"tamaño del vocabulario: {long_vocab}")

# Autogradiente para aplicar de forma recursiva la regla de la cadena a través de un grafo de cómputo
class Valor:
    __slots__ = ('dato', 'gradiente', '_hijos', '_grads_locales') # Optimización para el uso de memoria

    def __init__(auto, dato, hijos=(), grads_locales=()):
        auto.dato = dato                    # Valor escalar de este nodo (auto) calculado durante la propagación
        auto.gradiente = 0                  # Derivada de la pérdida con respecto a este nodo, calculada en la retropropagación
        auto._hijos = hijos                 # Hijos de este nodo en el grafo de cómputo
        auto._grads_locales = grads_locales # Derivada local de este nodo con respecto a sus hijos

    def __add__(auto, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro) # Si otro nodo es un objeto Valor, convertir en uno (ej: 5 -> Valor(5))
        return Valor(auto.dato + otro.dato, (auto, otro), (1, 1))

    def __mul__(auto, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro) # Garantizar que ambos operandos sean objetos Valor
        return Valor(auto.dato * otro.dato, (auto, otro), (otro.dato, auto.dato))
    
    # Operadores derivados
    def __pow__(auto, exponente): return Valor(auto.dato**exponente, (auto,), (exponente * auto.dato**(exponente-1),)) # Exponenciación
    def log(auto): return Valor(math.log(auto.dato), (auto,), (1/auto.dato,)) # Logaritmo natural
    def exp(auto): return Valor(math.exp(auto.dato), (auto,), (math.exp(auto.dato),)) # Exponencial
    def relu(auto): return Valor(max(0, auto.dato), (auto,), (float(auto.dato > 0),)) # Unidad lineal rectificada
    def __neg__(auto): return auto * -1 # Negación unaria
    def __radd__(auto, otro): return auto + otro # Suma con operando invertido (derecha)
    def __sub__(auto, otro): return auto + (-otro) # Resta
    def __rsub__(auto, otro): return otro + (-auto) # Resta con operando invertido
    def __rmul__(auto, otro): return auto * otro # Multiplicación con operando invertido
    def __truediv__(auto, otro): return auto * otro**-1 # División
    def __rtruediv__(auto, otro): return otro * auto**-1 # División con operando invertido

    def propagar(auto):
        orden_topologico = []
        visitados = set()
        def construir_orden(v):
            if v not in visitados:
                visitados.add(v)
                for hijo in v._hijos:
                    construir_orden(hijo)
                orden_topologico.append(v)
        construir_orden(auto)
        auto.gradiente = 1
        for v in reversed(orden_topologico):
            for hijo, grad_local in zip(v._hijos, v._grads_locales):
                hijo.gradiente += grad_local * v.gradiente

# Arrancar los parámetros para almacenar el conocimiento del modelo
c_capas = 1                             # Cantidad de capas
d_incrustacion = 16                     # Dimensión de incrustación
t_bloque = 16                           # Longitud máxima de secuencia
c_cabezas = 4                           # Cantidad de cabezas de atención
d_cabeza = d_incrustacion // c_cabezas  # Dimensión derivada de cada cabeza
matriz = lambda salidas, entradas, desv=0.08: [[Valor(random.gauss(0, desv)) for _ in range(entradas)] for _ in range(salidas)]
estado_modelo = {'incrustacion_fichas': matriz(long_vocab, d_incrustacion), 'incrustacion_posicion': matriz(t_bloque, d_incrustacion), 'cabeza_salida': matriz(long_vocab, d_incrustacion)}
for i in range(c_capas):
    estado_modelo[f'capa{i}.atencion_wq'] = matriz(d_incrustacion, d_incrustacion)
    estado_modelo[f'capa{i}.atencion_wk'] = matriz(d_incrustacion, d_incrustacion)
    estado_modelo[f'capa{i}.atencion_wv'] = matriz(d_incrustacion, d_incrustacion)
    estado_modelo[f'capa{i}.atencion_wo'] = matriz(d_incrustacion, d_incrustacion)
    estado_modelo[f'capa{i}.pmc_fc1'] = matriz(4 * d_incrustacion, d_incrustacion)
    estado_modelo[f'capa{i}.pmc_fc2'] = matriz(d_incrustacion, 4 * d_incrustacion)
parametros = [pa for matriz in estado_modelo.values() for fila in matriz for pa in fila] # Aplanar los parámetros en una sola lista[Valor]
print(f"cantidad de parámetros: {len(parametros)}")

# Definir la arquitectura del modelo: función que relaciona fichas y parámetros con logits (predicciones) sobre la próxima ficha
# Replica GPT-2, con diferencias menores: normalización RCM en lugar de normalización por capas, sin sesgos, ReLU en lugar de GeLU
def lineal(entrada, pesos):
    return [sum(peso_i * x_i for peso_i, x_i in zip(fila_pesos, entrada)) for fila_pesos in pesos]

def softmax(logits):
    valor_maximo = max(valor.dato for valor in logits)
    exponenciales = [(valor - valor_maximo).exp() for valor in logits]
    total = sum(exponenciales)
    return [e / total for e in exponenciales]

def normar_rcm(x):
    media_cuadratica = sum(x_i * x_i for x_i in x) / len(x)
    escala = (media_cuadratica + 1e-5) ** -0.5
    return [x_i * escala for x_i in x] # Nota: no es redundante debido a la retropropagación mediante la conexión residual

def gpt(id_ficha, id_posicion, claves, valores):
    incrustacion_ficha = estado_modelo['incrustacion_fichas'][id_ficha] # Incrustación de ficha
    incrustacion_posicion = estado_modelo['incrustacion_posicion'][id_posicion] # Incrustación de posición
    x = [f + p for f, p in zip(incrustacion_ficha, incrustacion_posicion)] # Combinar información semántica y posicional
    x = normar_rcm(x)

    for li in range(c_capas):
        # 1) Bloque de atención multicabeza
        x_residual = x
        x = normar_rcm(x)
        q = lineal(x, estado_modelo[f'capa{li}.atencion_wq'])
        k = lineal(x, estado_modelo[f'capa{li}.atencion_wk'])
        v = lineal(x, estado_modelo[f'capa{li}.atencion_wv'])
        claves[li].append(k)
        valores[li].append(v)
        x_atencion = []
        for h in range(c_cabezas):
            inicio = h * d_cabeza
            q_h = q[inicio:inicio+d_cabeza]
            k_h = [ki[inicio:inicio+d_cabeza] for ki in claves[li]]
            v_h = [vi[inicio:inicio+d_cabeza] for vi in valores[li]]
            logits_atencion = [sum(q_h[j] * k_h[t][j] for j in range(d_cabeza)) / d_cabeza**0.5 for t in range(len(k_h))]
            pesos_atencion = softmax(logits_atencion)
            salida_cabeza = [sum(pesos_atencion[t] * v_h[t][j] for t in range(len(v_h))) for j in range(d_cabeza)]
            x_atencion.extend(salida_cabeza)
        x = lineal(x_atencion, estado_modelo[f'capa{li}.atencion_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) Bloque PMC (Perceptrón Multicapa)
        x_residual = x
        x = normar_rcm(x)
        x = lineal(x, estado_modelo[f'capa{li}.pmc_fc1'])
        x = [xi.relu() for xi in x]
        x = lineal(x, estado_modelo[f'capa{li}.pmc_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = lineal(x, estado_modelo['cabeza_salida'])
    return logits

# Configurar optimizador Adam y su memoria
tasa_aprendizaje, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
momento1 = [0.0] * len(parametros) # Memoria de primer momento
momento2 = [0.0] * len(parametros) # Memoria de segundo momento

# Bucle de entrenamiento
c_pasos = 1000 # Cantidad de pasos de entrenamiento
for paso in range(c_pasos):
    # Tomar un documento, transformarlo en ficha y rodearlo con la ficha especial PDS en ambos lados
    doc = documentos[paso % len(documentos)]
    fichas = [PDS] + [caracteres_unicos.index(ch) for ch in doc] + [PDS]
    n = min(t_bloque, len(fichas) - 1)
    # Propagar la secuencia de fichas a través del modelo, construyendo el grafo de cómputo hasta la pérdida
    claves, valores = [[] for _ in range(c_capas)], [[] for _ in range(c_capas)]
    perdidas = []
    for pos_id in range(n):
        ficha_actual, ficha_objetivo = fichas[pos_id], fichas[pos_id + 1]
        logits = gpt(ficha_actual, pos_id, claves, valores)
        probabilidades = softmax(logits)
        perdida_pos = -probabilidades[ficha_objetivo].log()
        perdidas.append(perdida_pos)
    perdida = (1 / n) * sum(perdidas) # Pérdida promedio sobre la secuencia
    # Retropropagar la pérdida, calculando los gradientes con respecto a todos los parámetros
    perdida.propagar()

    # Actualización del optimizador Adam: actualiza los parámetros del modelo según los gradientes correspondientes
    tasa_actual = tasa_aprendizaje * (1 - paso / c_pasos) # Decaimiento lineal de la tasa de aprendizaje
    for i, p in enumerate(parametros):
        momento1[i] = beta1 * momento1[i] + (1 - beta1) * p.gradiente
        momento2[i] = beta2 * momento2[i] + (1 - beta2) * p.gradiente ** 2
        momento1_corregido = momento1[i] / (1 - beta1 ** (paso + 1))
        momento2_corregido = momento2[i] / (1 - beta2 ** (paso + 1))
        p.dato -= tasa_actual * momento1_corregido / (momento2_corregido ** 0.5 + eps_adam)
        p.gradiente = 0
    print(f"paso {paso+1:4d} / {c_pasos:4d} | pérdida {perdida.dato:.4f}", end="\r")

# Inferencia: generar texto nuevo
temperatura = 0.5 # En (0, 1], controla la «creatividad» del texto generado, de menor a mayor
print("\n--- inferencia (nuevos nombres alucinados) ---")
for ind_muestra in range(20):
    claves, valores = [[] for _ in range(c_capas)], [[] for _ in range(c_capas)]
    ficha_actual = PDS
    muestra = []
    for pos_id in range(t_bloque):
        logits = gpt(ficha_actual, pos_id, claves, valores)
        probabilidades = softmax([l / temperatura for l in logits])
        ficha_actual = random.choices(range(long_vocab), weights=[p.dato for p in probabilidades])[0]
        if ficha_actual == PDS:
            break
        muestra.append(caracteres_unicos[ficha_actual])
    print(f"muestra {ind_muestra+1:2d}: {''.join(muestra)}")