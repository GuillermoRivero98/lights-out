"""
Resolución del juego Lights Out mediante sistemas lineales sobre {0, 1}.

Modelo (según la letra del proyecto):
  * El tablero es una matriz n x n con entradas 1 (luz prendida) y 0 (apagada).
  * Una solución es un vector x = (x_1, ..., x_{n^2}) en {0, 1}^{n^2}. Para cada
    i se escribe i - 1 = n*q + r con 0 <= r < n, y x_i = 1 si y solo si la luz
    a_{(q+1)(r+1)} se presiona.
  * La ecuación i del sistema cuenta los cambios de estado de la luz
    a_{(q+1)(r+1)}: la luz misma y sus vecinas ortogonales. El término
    independiente es el estado inicial de esa luz (hay que cambiarla si y solo
    si está prendida).
  * Se trabaja con la suma binaria (1 + 1 = 0), que en Python es el XOR (^).

En el código los índices son base 0: la incógnita k = i - 1 corresponde a la
posición (fila, columna) = divmod(k, n), que es (q, r).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


# Desplazamientos (fila, columna) de una luz y sus cuatro vecinas ortogonales.
VECINDARIO = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# Indexación
# ---------------------------------------------------------------------------

def indice_a_posicion(i: int, n: int) -> tuple[int, int]:
    """Convierte el índice i (1..n^2) de la letra en la posición (fila, columna)
    de la luz, también con numeración desde 1. Usa i - 1 = n*q + r."""
    if not 1 <= i <= n * n:
        raise ValueError(f"El índice debe estar entre 1 y {n * n}.")
    q, r = divmod(i - 1, n)
    return q + 1, r + 1


def posicion_a_indice(fila: int, columna: int, n: int) -> int:
    """Operación inversa de indice_a_posicion (numeración desde 1)."""
    return n * (fila - 1) + columna


# ---------------------------------------------------------------------------
# Tableros de ejemplo de la consigna
# ---------------------------------------------------------------------------

# Tablero 3 x 3 resuelto en la consigna (solución: presionar a22 y a33).
TABLERO_CONSIGNA_3X3 = np.array([[0, 1, 0],
                                 [1, 1, 0],
                                 [0, 0, 1]], dtype=np.uint8)

# Tablero 5 x 5 de la consigna, antes y después de presionar a43.
TABLERO_CONSIGNA_5X5 = np.array([[0, 0, 1, 1, 1],
                                 [0, 1, 1, 1, 0],
                                 [1, 0, 0, 1, 1],
                                 [1, 1, 0, 0, 1],
                                 [0, 1, 1, 1, 0]], dtype=np.uint8)
TABLERO_CONSIGNA_5X5_TRAS_A43 = np.array([[0, 0, 1, 1, 1],
                                          [0, 1, 1, 1, 0],
                                          [1, 0, 1, 1, 1],
                                          [1, 0, 1, 1, 1],
                                          [0, 1, 0, 1, 0]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Simulación del juego (independiente del álgebra, sirve para verificar)
# ---------------------------------------------------------------------------

def presionar(tablero: np.ndarray, fila: int, columna: int) -> np.ndarray:
    """Devuelve el tablero que resulta de presionar la luz (fila, columna),
    con índices base 0. Cambia de estado la luz y sus vecinas ortogonales."""
    nuevo = tablero.copy()
    n = nuevo.shape[0]
    for df, dc in VECINDARIO:
        f, c = fila + df, columna + dc
        if 0 <= f < n and 0 <= c < n:
            nuevo[f, c] ^= 1
    return nuevo


def aplicar_solucion(tablero: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Presiona todas las luces marcadas con 1 en el vector x y devuelve el
    tablero final."""
    n = tablero.shape[0]
    resultado = np.asarray(tablero, dtype=np.uint8).copy()
    for k in np.flatnonzero(x):
        fila, columna = divmod(int(k), n)
        resultado = presionar(resultado, fila, columna)
    return resultado


def es_solucion(tablero: np.ndarray, x: np.ndarray) -> bool:
    """True si presionar las luces indicadas por x apaga todo el tablero."""
    return not aplicar_solucion(tablero, x).any()


# ---------------------------------------------------------------------------
# Construcción del sistema lineal
# ---------------------------------------------------------------------------

def matriz_del_sistema(n: int) -> np.ndarray:
    """Matriz A (n^2 x n^2) del sistema A x = b.

    A[k, m] = 1 si presionar la luz m cambia el estado de la luz k, es decir,
    si m = k o si las luces k y m son adyacentes (arriba, abajo, izquierda o
    derecha). La matriz es simétrica: la adyacencia es una relación simétrica.
    """
    A = np.zeros((n * n, n * n), dtype=np.uint8)
    for k in range(n * n):
        fila, columna = divmod(k, n)
        for df, dc in VECINDARIO:
            f, c = fila + df, columna + dc
            if 0 <= f < n and 0 <= c < n:
                A[k, f * n + c] = 1
    return A


def vector_independiente(tablero: np.ndarray) -> np.ndarray:
    """Vector b: la entrada k es el estado inicial de la luz k (recorriendo el
    tablero por filas, igual que las incógnitas)."""
    return np.asarray(tablero, dtype=np.uint8).reshape(-1) % 2


# ---------------------------------------------------------------------------
# Escalerización gaussiana sobre {0, 1}
# ---------------------------------------------------------------------------

@dataclass
class Escalerizacion:
    """Resultado de escalerizar la matriz ampliada [A | b]."""
    ampliada: np.ndarray      # matriz escalerizada reducida [R | c]
    pivotes: list[int]        # columna del pivote de cada fila no nula
    correcciones: int         # veces que el pivote era 0 y se le sumó otra fila

    @property
    def rango(self) -> int:
        return len(self.pivotes)

    @property
    def es_compatible(self) -> bool:
        """El sistema es compatible si ninguna fila queda 0 = 1."""
        return not self.ampliada[self.rango:, -1].any()

    @property
    def variables_libres(self) -> list[int]:
        total = self.ampliada.shape[1] - 1
        con_pivote = set(self.pivotes)
        return [j for j in range(total) if j not in con_pivote]


def escalerizar_gf2(A: np.ndarray, b: np.ndarray) -> Escalerizacion:
    """Escalerización de Gauss-Jordan de [A | b] con suma binaria.

    En {0, 1} un pivote no nulo vale 1, por lo que no hace falta multiplicar
    filas, y la única transformación que se usa es F_i -> F_i + F_j (XOR).

    Si la entrada en la posición de pivote es 0 (en el informe se muestra que
    esto ocurre ya en el segundo paso), no se intercambian filas: a la fila
    del pivote se le suma una fila inferior que tenga un 1 en esa columna, lo
    que también es una transformación del tipo F_i -> F_i + F_j.
    """
    M = np.concatenate([A % 2, (b % 2)[:, None]], axis=1).astype(np.uint8)
    filas, columnas = A.shape
    pivotes: list[int] = []
    correcciones = 0
    fila_actual = 0

    for col in range(columnas):
        if fila_actual == filas:
            break
        # Primera fila (desde la actual) con un 1 en esta columna.
        candidatas = np.flatnonzero(M[fila_actual:, col])
        if candidatas.size == 0:
            continue  # no hay pivote: la variable de esta columna es libre
        p = fila_actual + int(candidatas[0])
        if p != fila_actual:
            # Pivote nulo: F_pivote -> F_pivote + F_p. La fila p está debajo,
            # así que tiene ceros en las columnas de los pivotes anteriores y
            # no los altera.
            M[fila_actual] ^= M[p]
            correcciones += 1
        # F_i -> F_i + F_pivote en todas las demás filas con un 1 en la columna
        # (arriba y abajo del pivote, para obtener la forma reducida).
        con_uno = np.flatnonzero(M[:, col])
        con_uno = con_uno[con_uno != fila_actual]
        M[con_uno] ^= M[fila_actual]
        pivotes.append(col)
        fila_actual += 1

    return Escalerizacion(M, pivotes, correcciones)


def base_del_nucleo(esc: Escalerizacion) -> list[np.ndarray]:
    """Base del núcleo de A a partir de la forma escalerizada reducida.

    Cada variable libre da un vector: vale 1 en esa variable, 0 en las demás
    libres, y cada variable pivote toma el valor que anula su ecuación.
    Estos vectores son combinaciones de pulsaciones que no cambian el tablero.
    """
    n2 = esc.ampliada.shape[1] - 1
    base = []
    for libre in esc.variables_libres:
        v = np.zeros(n2, dtype=np.uint8)
        v[libre] = 1
        for fila, piv in enumerate(esc.pivotes):
            v[piv] = esc.ampliada[fila, libre]
        base.append(v)
    return base


# ---------------------------------------------------------------------------
# Función pedida en la letra
# ---------------------------------------------------------------------------

class SinSolucionError(ValueError):
    """El tablero no puede apagarse por completo."""


def resolver_lights_out(tablero, minimizar: bool = False) -> np.ndarray:
    """Recibe una matriz n x n de unos y ceros (estado inicial) y devuelve un
    vector de unos y ceros de largo n^2 con las luces a presionar, según el
    modelo de la letra.

    Si el sistema tiene varias soluciones se devuelve la que asigna 0 a todas
    las variables libres; con minimizar=True se devuelve una de las que
    requieren la menor cantidad de pulsaciones (recorre todas las soluciones,
    por eso solo se permite si hay a lo sumo 2^20 de ellas).

    Lanza SinSolucionError si el tablero no tiene solución.
    """
    tablero = np.asarray(tablero)
    if tablero.ndim != 2 or tablero.shape[0] != tablero.shape[1]:
        raise ValueError("El tablero debe ser una matriz cuadrada n x n.")
    # Se valida antes de convertir el tipo, para no aceptar valores como 0.5.
    if not np.isin(tablero, (0, 1)).all():
        raise ValueError("El tablero solo puede contener ceros y unos.")
    tablero = tablero.astype(np.uint8)

    n = tablero.shape[0]
    esc = escalerizar_gf2(matriz_del_sistema(n), vector_independiente(tablero))
    if not esc.es_compatible:
        raise SinSolucionError("Este estado del tablero no tiene solución.")

    # Solución particular: variables libres en 0, cada pivote toma el valor
    # del término independiente de su fila.
    x = np.zeros(n * n, dtype=np.uint8)
    for fila, piv in enumerate(esc.pivotes):
        x[piv] = esc.ampliada[fila, -1]

    if minimizar:
        base = base_del_nucleo(esc)
        if len(base) > 20:
            raise ValueError("Demasiadas soluciones para recorrerlas todas.")
        mejor = x
        for coeficientes in product((0, 1), repeat=len(base)):
            candidata = x.copy()
            for c, v in zip(coeficientes, base):
                if c:
                    candidata ^= v
            if candidata.sum() < mejor.sum():
                mejor = candidata
        x = mejor

    return x


def primer_pivote_nulo(n: int) -> int | None:
    """Escaleriza la matriz del sistema n x n tomando siempre la fila k tal
    como está como fila del pivote en el paso k, y aplicando F_i -> F_i + F_k
    a las filas de abajo con un 1 en la columna k (sin ninguna corrección del
    pivote).

    Devuelve el primer paso k (numerado desde 1) en el que la entrada (k, k)
    es 0, es decir, en el que el método sin pivoteo no puede continuar, o None
    si eso nunca ocurre.
    """
    M = matriz_del_sistema(n)
    for k in range(n * n):
        if M[k, k] == 0:
            return k + 1
        abajo = k + 1 + np.flatnonzero(M[k + 1:, k])
        M[abajo] ^= M[k]
    return None
