# Actividad 7: supervivencia en el Titanic

Se requiere Python y el archivo `Titanic-Dataset.csv` en esta misma carpeta.
En PowerShell, desde esta carpeta:

En esta computadora se instalaron las dependencias en `.dependencias`. Para usar esa instalación local:

```powershell
$env:PYTHONPATH = (Resolve-Path .\.dependencias).Path
python actividad7.py
```

En otra computadora, instala primero las dependencias y ejecuta:

```powershell
python -m pip install -r requirements.txt
python actividad7.py
```

El programa guarda resultados en la carpeta `resultados`: gráficas, métricas, coeficientes, probabilidades de cada pasajero de prueba y un resumen. Al ejecutarlo nuevamente, esos archivos se actualizan.

El modelo base utiliza sexo, clase y edad. El ampliado agrega tarifa y número de hermanos/cónyuges a bordo. La separación 80/20 es la misma y está estratificada por supervivencia. Las medianas y la codificación se ajustan con entrenamiento, sin usar prueba. Las probabilidades y métricas corresponden a la partición de prueba; se comparan los umbrales 0.5 y 0.7.

En los coeficientes, `Sex_male` compara hombres con mujeres, que es la categoría de referencia. Una razón de odds de 1 no cambia las odds; mayor que 1 las multiplica y menor que 1 las reduce. Las variables tienen unidades distintas, por lo que sus coeficientes no deben compararse directamente para elegir la más importante. Las asociaciones no demuestran causalidad.
