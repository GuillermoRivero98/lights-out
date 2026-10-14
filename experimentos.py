"""
Experimentos del informe. Genera las figuras y las tablas (en formato LaTeX)
que se incluyen en el informe, a partir de lights_out.py.

Uso:  python experimentos.py
Salida: carpeta informe/generado/ (figuras .pdf, tablas .tex y resultados.json)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from lights_out import (
    TABLERO_CONSIGNA_3X3 as TABLERO_LETRA_3X3,
    TABLERO_CONSIGNA_5X5 as TABLERO_LETRA_5X5,
    SinSolucionError, aplicar_solucion, base_del_nucleo, es_solucion,
    escalerizar_gf2, matriz_del_sistema, primer_pivote_nulo,
    resolver_lights_out, vector_independiente,
)

SEMILLA = 2026            # una única semilla para todo el experimento
TABLEROS_POR_N = 2000     # tableros aleatorios por tamaño
TAMANOS = range(1, 11)    # n = 1, ..., 10
SALIDA = Path(__file__).parent / "informe" / "generado"

# Captura del juego Grid Lights (verde = prendida = 1, rojo = apagada = 0).
TABLERO_GRID_LIGHTS = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])

# Colores (paleta de referencia de la guía de visualización).
AZUL, AMARILLO = "#2a78d6", "#eda100"
GRIS_CLARO, TINTA, TINTA_2 = "#e4e3df", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.edgecolor": TINTA_2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def analizar_tamano(n: int, rng: np.random.Generator) -> dict:
    """Propiedades de la matriz del sistema n x n y prueba con tableros
    aleatorios: cuántos tienen solución y si todas las soluciones devueltas
    apagan el tablero al simularlas."""
    A = matriz_del_sistema(n)
    esc = escalerizar_gf2(A, np.zeros(n * n, dtype=np.uint8))
    rango = esc.rango
    nucleo = n * n - rango

    con_solucion = verificadas = 0
    for _ in range(TABLEROS_POR_N):
        tablero = rng.integers(0, 2, (n, n))
        try:
            x = resolver_lights_out(tablero)
        except SinSolucionError:
            continue
        con_solucion += 1
        verificadas += es_solucion(tablero, x)

    return {
        "n": n,
        "incognitas": n * n,
        "rango": rango,
        "dim_nucleo": nucleo,
        "fraccion_teorica": 2.0 ** -nucleo,
        "fraccion_empirica": con_solucion / TABLEROS_POR_N,
        "con_solucion": con_solucion,
        "verificadas": verificadas,
        "paso_pivote_nulo": primer_pivote_nulo(n),
        "correcciones_de_pivote": esc.correcciones,
    }


def comparar_pulsaciones(n: int, cantidad: int, rng: np.random.Generator):
    """Promedio de pulsaciones de la solución con variables libres en 0 frente
    a la de mínima cantidad de pulsaciones, en tableros con solución.

    Los tableros con solución se generan como A z (mod 2) para un vector z
    aleatorio: así se obtienen tableros uniformes dentro de los que tienen
    solución, sin descartar intentos (para n = 9 solo 1 de cada 256 la tiene).
    """
    A = matriz_del_sistema(n)
    particular, minima = [], []
    for _ in range(cantidad):
        z = rng.integers(0, 2, n * n)
        tablero = (A @ z % 2).reshape(n, n)
        x = resolver_lights_out(tablero)
        x_min = resolver_lights_out(tablero, minimizar=True)
        if not es_solucion(tablero, x_min):
            raise RuntimeError("La solución mínima no apaga el tablero.")
        particular.append(int(x.sum()))
        minima.append(int(x_min.sum()))
    return float(np.mean(particular)), float(np.mean(minima))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def dibujar_tablero(ax, tablero, pulsaciones=None, titulo=""):
    """Dibuja un tablero (amarillo = prendida, gris = apagada) y marca con un
    punto las luces que se presionan."""
    n = tablero.shape[0]
    ax.imshow(tablero, cmap=ListedColormap([GRIS_CLARO, AMARILLO]),
              vmin=0, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(range(1, n + 1))
    ax.set_yticklabels(range(1, n + 1))
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    if pulsaciones is not None:
        for k in np.flatnonzero(pulsaciones):
            f, c = divmod(int(k), n)
            ax.plot(c, f, "o", color=TINTA, markersize=9)
    ax.set_title(titulo, fontsize=10)


def figura_ejemplos(ruta: Path):
    """Los dos tableros 3 x 3 resueltos con la función."""
    fig, ejes = plt.subplots(2, 3, figsize=(6.4, 4.6))
    casos = [("Tablero de la consigna", TABLERO_LETRA_3X3),
             ("Captura de Grid Lights", TABLERO_GRID_LIGHTS)]
    for fila, (nombre, tablero) in enumerate(casos):
        x = resolver_lights_out(tablero)
        dibujar_tablero(ejes[fila, 0], tablero, titulo="Estado inicial")
        dibujar_tablero(ejes[fila, 1], tablero, x, titulo="Luces a presionar")
        dibujar_tablero(ejes[fila, 2], aplicar_solucion(tablero, x),
                        titulo="Estado final")
        ejes[fila, 0].set_ylabel(nombre, fontsize=10, labelpad=8)
    fig.tight_layout()
    fig.savefig(ruta, metadata={"CreationDate": None})
    plt.close(fig)


def figura_fraccion(resultados: list[dict], ruta: Path):
    """Fracción de tableros con solución, observada y teórica."""
    ns = [r["n"] for r in resultados]
    empirica = [r["fraccion_empirica"] for r in resultados]
    teorica = [r["fraccion_teorica"] for r in resultados]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(ns, empirica, width=0.55, color=AZUL,
           label=f"Observada ({TABLEROS_POR_N} tableros por $n$)")
    ax.plot(ns, teorica, "D", color=TINTA, markersize=6, linestyle="none",
            label=r"Teórica: $2^{-\dim N(A)}$")
    ax.set_xticks(ns)
    ax.set_xlabel(r"Tamaño del tablero $n$")
    ax.set_ylabel("Fracción con solución")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: coma(v, 1))
    ax.yaxis.grid(True, color=GRIS_CLARO, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, metadata={"CreationDate": None})
    plt.close(fig)


def figura_matriz(n: int, ruta: Path):
    """Patrón de unos de la matriz del sistema."""
    A = matriz_del_sistema(n)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(A, cmap=ListedColormap(["white", AZUL]), vmin=0, vmax=1)
    ticks = np.arange(0, n * n, n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)
    ax.set_xlabel(r"Incógnita $x_j$ (luz presionada)")
    ax.set_ylabel(r"Ecuación $i$ (luz afectada)")
    for lado in ax.spines.values():
        lado.set_visible(True)
    fig.tight_layout()
    fig.savefig(ruta, metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tablas en LaTeX
# ---------------------------------------------------------------------------

def coma(valor: float, decimales: int) -> str:
    """Formatea un número con coma decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def tabla_tamanos(resultados: list[dict], ruta: Path):
    filas = []
    for r in resultados:
        paso = r["paso_pivote_nulo"] if r["paso_pivote_nulo"] else "---"
        filas.append(
            f"{r['n']} & {r['incognitas']} & {r['rango']} & {r['dim_nucleo']}"
            f" & {coma(r['fraccion_teorica'], 4)}"
            f" & {coma(r['fraccion_empirica'], 4)}"
            f" & {r['verificadas']}/{r['con_solucion']} & {paso}"
            f" & {r['correcciones_de_pivote']} \\\\")
    ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")


def tabla_pulsaciones(filas: list[tuple], ruta: Path):
    lineas = [f"{n} & {d} & {cant} & {coma(p, 3)} & {coma(m, 3)} \\\\"
              for n, d, cant, p, m in filas]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)

    resultados = [analizar_tamano(n, rng) for n in TAMANOS]

    pulsaciones = []
    for n in (4, 5, 9):
        d = next(r["dim_nucleo"] for r in resultados if r["n"] == n)
        p, m = comparar_pulsaciones(n, 200, rng)
        pulsaciones.append((n, d, 200, p, m))

    x_letra = resolver_lights_out(TABLERO_LETRA_3X3)
    x_grid = resolver_lights_out(TABLERO_GRID_LIGHTS)
    esc_5x5 = escalerizar_gf2(matriz_del_sistema(5),
                              vector_independiente(TABLERO_LETRA_5X5))
    # Producto del tablero 5 x 5 con cada vector de la base del núcleo: el
    # tablero tiene solución si y solo si todos estos productos son 0.
    nucleo_5x5 = base_del_nucleo(escalerizar_gf2(
        matriz_del_sistema(5), np.zeros(25, dtype=np.uint8)))
    productos_5x5 = [int(v @ vector_independiente(TABLERO_LETRA_5X5) % 2)
                     for v in nucleo_5x5]

    figura_ejemplos(SALIDA / "figura_ejemplos.pdf")
    figura_fraccion(resultados, SALIDA / "figura_fraccion.pdf")
    figura_matriz(4, SALIDA / "figura_matriz_4x4.pdf")
    tabla_tamanos(resultados, SALIDA / "tabla_tamanos.tex")
    tabla_pulsaciones(pulsaciones, SALIDA / "tabla_pulsaciones.tex")

    resumen = {
        "semilla": SEMILLA,
        "tableros_por_n": TABLEROS_POR_N,
        "solucion_letra_3x3": x_letra.tolist(),
        "solucion_grid_lights": x_grid.tolist(),
        "letra_5x5_tiene_solucion": bool(esc_5x5.es_compatible),
        "letra_5x5_productos_con_nucleo": productos_5x5,
        "base_nucleo_5x5": [v.reshape(5, 5).tolist() for v in nucleo_5x5],
        "por_tamano": resultados,
        "pulsaciones": [dict(zip(("n", "dim_nucleo", "tableros",
                                  "promedio_particular", "promedio_minima"), f))
                        for f in pulsaciones],
    }
    (SALIDA / "resultados.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumen breve en consola.
    print(f"Solución tablero 3x3 de la letra: {x_letra.tolist()}")
    print(f"Solución captura Grid Lights:     {x_grid.tolist()}")
    print(f"Tablero 5x5 de la letra con solución: {esc_5x5.es_compatible}"
          f" (productos con la base del núcleo: {productos_5x5})")
    print(" n  rango  dim N(A)  teórica  empírica  verificadas")
    for r in resultados:
        print(f"{r['n']:2d}  {r['rango']:5d}  {r['dim_nucleo']:8d}  "
              f"{r['fraccion_teorica']:7.4f}  {r['fraccion_empirica']:8.4f}  "
              f"{r['verificadas']}/{r['con_solucion']}")
    print(f"Figuras y tablas guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
