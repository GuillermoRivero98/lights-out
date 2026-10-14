"""Pruebas de lights_out.py. Ejecutar con:  python -m unittest -v"""

import unittest
from itertools import product

import numpy as np

from lights_out import (
    TABLERO_CONSIGNA_3X3, TABLERO_CONSIGNA_5X5, TABLERO_CONSIGNA_5X5_TRAS_A43,
    SinSolucionError, posicion_a_indice, aplicar_solucion, base_del_nucleo, es_solucion,
    escalerizar_gf2, indice_a_posicion, matriz_del_sistema, presionar,
    primer_pivote_nulo, resolver_lights_out, vector_independiente,
)

TABLERO_3X3 = TABLERO_CONSIGNA_3X3
TABLERO_5X5 = TABLERO_CONSIGNA_5X5
TABLERO_5X5_TRAS_A43 = TABLERO_CONSIGNA_5X5_TRAS_A43

# Las nueve ecuaciones del sistema de la letra, escritas como listas de
# incógnitas (numeradas desde 1) y término independiente.
SISTEMA_LETRA = [
    ([1, 2, 4], 0), ([1, 2, 3, 5], 1), ([2, 3, 6], 0),
    ([1, 4, 5, 7], 1), ([2, 4, 5, 6, 8], 1), ([3, 5, 6, 9], 0),
    ([4, 7, 8], 0), ([5, 7, 8, 9], 0), ([6, 8, 9], 1),
]


class TestModelo(unittest.TestCase):

    def test_indexacion_de_la_letra(self):
        # i - 1 = n q + r  ->  luz a_{(q+1)(r+1)}
        self.assertEqual(indice_a_posicion(1, 3), (1, 1))
        self.assertEqual(indice_a_posicion(5, 3), (2, 2))
        self.assertEqual(indice_a_posicion(9, 3), (3, 3))
        self.assertEqual(indice_a_posicion(18, 5), (4, 3))
        # La correspondencia es biunívoca.
        for n in range(1, 7):
            for i in range(1, n * n + 1):
                self.assertEqual(posicion_a_indice(*indice_a_posicion(i, n), n), i)

    def test_jugada_a43_de_la_letra(self):
        # a_43 es la fila 4, columna 3 (base 0: (3, 2)).
        np.testing.assert_array_equal(presionar(TABLERO_5X5, 3, 2),
                                      TABLERO_5X5_TRAS_A43)

    def test_sistema_3x3_coincide_con_la_letra(self):
        A = matriz_del_sistema(3)
        b = vector_independiente(TABLERO_3X3)
        for i, (incognitas, termino) in enumerate(SISTEMA_LETRA):
            esperado = np.zeros(9, dtype=np.uint8)
            esperado[np.array(incognitas) - 1] = 1
            np.testing.assert_array_equal(A[i], esperado)
            self.assertEqual(b[i], termino)

    def test_matriz_simetrica(self):
        for n in range(1, 8):
            A = matriz_del_sistema(n)
            np.testing.assert_array_equal(A, A.T)

    def test_orden_y_repeticion_no_importan(self):
        rng = np.random.default_rng(1)
        t = rng.integers(0, 2, (4, 4))
        # Presionar dos veces la misma luz no cambia el tablero.
        np.testing.assert_array_equal(presionar(presionar(t, 1, 2), 1, 2), t)
        # Presionar en distinto orden da el mismo tablero.
        np.testing.assert_array_equal(presionar(presionar(t, 0, 0), 2, 3),
                                      presionar(presionar(t, 2, 3), 0, 0))


def soluciones_por_fuerza_bruta(tablero):
    """Todas las soluciones, probando los 2^(n^2) conjuntos de pulsaciones con
    la simulación del juego (no usa la matriz del sistema)."""
    n = tablero.shape[0]
    return [np.array(x, dtype=np.uint8)
            for x in product((0, 1), repeat=n * n)
            if es_solucion(tablero, np.array(x, dtype=np.uint8))]


class TestFuerzaBruta(unittest.TestCase):
    """Comparación con una búsqueda exhaustiva, independiente del álgebra."""

    def test_todos_los_tableros_3x3(self):
        # Para n = 3 hay una única solución por tablero.
        for estado in product((0, 1), repeat=9):
            t = np.array(estado, dtype=np.uint8).reshape(3, 3)
            (unica,) = soluciones_por_fuerza_bruta(t)
            np.testing.assert_array_equal(resolver_lights_out(t), unica)

    def test_existencia_de_solucion_4x4(self):
        # Tableros 4x4 con solución: los que se obtienen al aplicar algún
        # conjunto de pulsaciones al tablero apagado (las reglas son
        # reversibles: aplicarlas de nuevo lo vuelve a apagar).
        apagado = np.zeros((4, 4), dtype=np.uint8)
        alcanzables = {aplicar_solucion(apagado, np.array(x)).tobytes()
                       for x in product((0, 1), repeat=16)}
        self.assertEqual(len(alcanzables), 2 ** 12)
        rng = np.random.default_rng(11)
        for _ in range(300):
            t = rng.integers(0, 2, (4, 4)).astype(np.uint8)
            if t.tobytes() in alcanzables:
                self.assertTrue(es_solucion(t, resolver_lights_out(t)))
            else:
                with self.assertRaises(SinSolucionError):
                    resolver_lights_out(t)


class TestResolucion(unittest.TestCase):

    def test_solucion_de_la_letra(self):
        x = resolver_lights_out(TABLERO_3X3)
        np.testing.assert_array_equal(x, [0, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertTrue(es_solucion(TABLERO_3X3, x))

    def test_tableros_aleatorios(self):
        rng = np.random.default_rng(2026)
        for n in range(1, 9):
            A = matriz_del_sistema(n)
            for _ in range(50):
                t = rng.integers(0, 2, (n, n))
                b = vector_independiente(t)
                try:
                    x = resolver_lights_out(t)
                except SinSolucionError:
                    # Si no hay solución, el sistema debe ser incompatible.
                    self.assertFalse(escalerizar_gf2(A, b).es_compatible)
                    continue
                self.assertTrue(es_solucion(t, x))
                np.testing.assert_array_equal(A @ x % 2, b)

    def test_minimizar_no_empeora(self):
        rng = np.random.default_rng(7)
        resueltos = 0
        while resueltos < 20:
            t = rng.integers(0, 2, (5, 5))
            try:
                x = resolver_lights_out(t)
            except SinSolucionError:
                continue
            x_min = resolver_lights_out(t, minimizar=True)
            self.assertTrue(es_solucion(t, x_min))
            self.assertLessEqual(x_min.sum(), x.sum())
            resueltos += 1

    def test_sin_solucion(self):
        # Una sola luz prendida en una esquina del 4 x 4 no tiene solución.
        t = np.zeros((4, 4), dtype=int)
        t[0, 0] = 1
        with self.assertRaises(SinSolucionError):
            resolver_lights_out(t)

    def test_nucleo_no_cambia_el_tablero(self):
        for n in (4, 5, 9):
            esc = escalerizar_gf2(matriz_del_sistema(n),
                                  np.zeros(n * n, dtype=np.uint8))
            base = base_del_nucleo(esc)
            self.assertEqual(len(base), n * n - esc.rango)
            for v in base:
                vacio = np.zeros((n, n), dtype=np.uint8)
                self.assertFalse(aplicar_solucion(vacio, v).any())

    def test_criterio_del_nucleo(self):
        # Como A es simétrica, b tiene solución si y solo si b es ortogonal
        # (con la suma binaria) a todos los vectores del núcleo de A.
        rng = np.random.default_rng(3)
        for n in (4, 5):
            A = matriz_del_sistema(n)
            base = base_del_nucleo(escalerizar_gf2(
                A, np.zeros(n * n, dtype=np.uint8)))
            for _ in range(100):
                b = rng.integers(0, 2, n * n).astype(np.uint8)
                ortogonal = all(int(v @ b) % 2 == 0 for v in base)
                self.assertEqual(ortogonal, escalerizar_gf2(A, b).es_compatible)

    def test_entradas_invalidas(self):
        with self.assertRaises(ValueError):
            resolver_lights_out([[0, 1, 0]])
        with self.assertRaises(ValueError):
            resolver_lights_out([[0, 2], [1, 0]])
        with self.assertRaises(ValueError):
            resolver_lights_out([[0.5, 1], [1, 1]])
        with self.assertRaises(ValueError):
            resolver_lights_out([[-1, 0], [0, 0]])

    def test_pivote_nulo_en_el_paso_2(self):
        # Tomando cada fila tal como está, el segundo pivote es 0.
        for n in range(2, 11):
            self.assertEqual(primer_pivote_nulo(n), 2)
        self.assertIsNone(primer_pivote_nulo(1))


if __name__ == "__main__":
    unittest.main()
