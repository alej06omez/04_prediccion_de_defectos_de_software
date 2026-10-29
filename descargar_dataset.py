"""Descarga NASA PC1 (OpenML 1068 / PROMISE) y ejecuta ETL."""
from backend.etl import descargar, preparar
from backend.paths import DATA_RAW, RAW_CSV


def main():
    print("Dataset: NASA PC1 software defect prediction")
    print("OpenML: https://www.openml.org/d/1068")
    print("Fuente academica PROMISE / NASA MDP. Revise licencia antes de uso fuera del curso.")
    path = descargar()
    df = preparar(path)
    print("CSV crudo:", path)
    print("Procesado:", len(df), "modulos;", int(df["defects"].sum()), "con defecto")
    print("Carpeta crudos:", DATA_RAW)


if __name__ == "__main__":
    main()
