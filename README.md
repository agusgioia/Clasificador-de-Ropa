# Clasificador-de-Ropa

# Resumen

Se utilizó un dataset de imágenes normalizadas a 28 x 28 en blanco y negro correspondientes a prendas de ropa para clasificarlas en 10 categorías. Se utilizaron dos modelos distintos, uno con una red neuronal densa y otro con una red neuronal convolucional para comparar resultados y sacar conclusiones.

## Archivos incluidos

* `ClasificadorRopaConvolucional.ipynb` — Red convolucional (CNN).
* `RopaRedDensa.ipynb` — Red totalmente conectada (Densa).
---

## Dataset

* **Fashion MNIST**
  
* Preprocesamiento idéntico en ambos notebooks: normalización `imagen /= 255`, reshape a `28×28` y batching (`TAMANO_LOTE = 32`).
* División estándar: 60.000 ejemplos entrenamiento, 10.000 pruebas.

---

## Hipótesis de entrenamiento comunes

* **Optimizer:** `Adam`.
* **Loss:** `SparseCategoricalCrossentropy`.
* **Métrica:** `accuracy`.
* **Epochs:** 30.
* **Batch size (TAMANO_LOTE):** 32.
* Los valores finales de accuracy/val_accuracy se obtuvieron de los logs impresos en cada notebook (última época).

---

## Modelo A — `ClasificadorRopaConvolucional.ipynb` (CNN)

**Arquitectura:**

* `Conv2D(32, 3x3, padding='same', activation='relu', input_shape=(28,28,1))`
* `MaxPool2D(2,2)`
* `Conv2D(64, 3x3, relu)` + `MaxPool2D`
* `Conv2D(64, 3x3, relu)` + `MaxPool2D`
* `Conv2D(128, 3x3, relu)` + `MaxPool2D`
* `Dropout(0.3)`
* `Flatten()`
* `Dense(100, relu)`
* `Dense(10, softmax)`

**Parámetros aproximados (calculado a partir de la arquitectura):** ~**143.510** parámetros entrenables.

**Resultados:**

* **Train accuracy:** **97.50%**
* **Validation accuracy:** **91.55%**

Observación: gap train/val ≈ 0.06 (indica algo de sobreajuste pero buena generalización relativa).

---

## Modelo B — `RopaRedDensa.ipynb` (Red densa)

**Arquitectura (resumen):**

* `Flatten(input_shape=(28,28,1))` → vector 784
* `Dense(50, relu)`
* `Dense(50, relu)`
* `Dense(10, softmax)`

**Parámetros aproximados:** ~**42.310** parámetros entrenables.

**Resultados (última época de entrenamiento según logs):**

* **Train accuracy:** **93.01%**
* **Validation accuracy:** **88.27%**

Observación: menor capacidad (menos parámetros) y rendimiento inferior en validación respecto a la CNN.

---

## Tabla Comparativa

| Modelo                                      | Parámetros aprox. | Train acc (final) | Val acc (final) |
| ------------------------------------------- | ----------------: | ----------------: | --------------: |
| CNN (`ClasificadorRopaConvolucional.ipynb`) |           143.510 |            97.50% |      **91.55%** |
| Densa (`RopaRedDensa.ipynb`)                |            42.310 |            93.01% |          88.27% |

---

## Conclusión

La **CNN** (`ClasificadorRopaConvolucional.ipynb`) obtuvo el **mejor rendimiento** en este experimento: **val_accuracy ≈ 91.6%** vs **88.3%** de la red densa. Tiene más parámetros (≈143k vs ≈42k) pero captura mejor las características espaciales de las imágenes.

