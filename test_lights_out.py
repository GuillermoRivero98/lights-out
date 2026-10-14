"""Pruebas de lights_out.py. Ejecutar con:  python -m unittest -v"""

import unittest

import numpy as np

from lights_out import (
    SinSolucionError, aplicar_solucion, base_del_nucleo, es_solucion,
    escalerizar_gf2, indice_a_posicion, matriz_del_sistema, presionar,
    primer_pivote_nulo, resolver_lights_out, vector_independiente,
)

# Ejemplos tomados de la letra del proyecto.
TABLERO_3X3 = np.array([[0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1]])
TABLERO_5X5 = np.array([[0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1],
                        [0, 1, 1, 1, 0]])
TABLERO_5X5_TRAS_A43 = np.array([[0, 0, 1, 1, 1],
                                 [0, 1, 1, 1, 0],
                                 [1, 0, 1, 1, 1],
                                 [1, 0, 1, 1, 1],
                                 [0, 1, 0, 1, 0]])

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

    def test_sin_pivoteo_falla(self):
        # La escalerización sin intercambiar filas se detiene en el paso 2.
        for n in range(2, 11):
            self.assertEqual(primer_pivote_nulo(n), 2)
        self.assertIsNone(primer_pivote_nulo(1))


if __name__ == "__main__":
    unittest.main()
