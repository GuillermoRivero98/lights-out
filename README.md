# Lights Out con álgebra lineal sobre {0, 1}

Proyecto del curso Álgebra Aplicada (Facultad de Ingeniería y Tecnologías, UCU).
Autores: Guillermo Rivero y Enzo Barreto.

El juego se modela como un sistema lineal `A x = b` con suma binaria
(1 + 1 = 0) y se resuelve con escalerización de Gauss-Jordan adaptada a esa
suma, con intercambio de filas.

## Archivos

| Archivo | Contenido |
|---|---|
| `lights_out.py` | Modelo, simulación del juego y la función `resolver_lights_out(tablero)` |
| `test_lights_out.py` | Pruebas automáticas (sistema de la consigna, jugada sobre a43, tableros aleatorios) |
| `experimentos.py` | Genera las tablas y figuras del informe en `informe/generado/` |
| `informe/informe.tex` | Fuente LaTeX del informe (la captura de Grid Lights está en `informe/imagenes/`) |

## Uso

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt

python -m unittest -v        # pruebas
python experimentos.py       # tablas, figuras y resultados.json (semilla 2026)
```

Ejemplo de uso de la función:

```python
from lights_out import resolver_lights_out, SinSolucionError

tablero = [[0, 1, 0],
           [1, 1, 0],
           [0, 0, 1]]
x = resolver_lights_out(tablero)
print(x)   # [0 0 0 0 1 0 0 0 1] -> presionar a22 y a33
```

Si el tablero no tiene solución, la función lanza `SinSolucionError`.
Con `minimizar=True` devuelve una de las soluciones con menos pulsaciones.

## Compilar el informe

```bash
cd informe
latexmk -pdf informe.tex
```
