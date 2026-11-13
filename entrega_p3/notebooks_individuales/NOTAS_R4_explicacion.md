# Notas de estudio del rol R4 (Evaluación y validación) — Práctica 2

Guía para que todo el equipo entienda **qué hace cada parte** de `notebooks/R4_evaluacion.ipynb`,
**por qué** se hace así y **cómo defenderlo**. Es material de apoyo del equipo, no un entregable de la práctica.

Todas las cifras de este documento salen de ejecutar el notebook con `PROJECT_RANDOM_STATE = 2627`.

---

## 0. La idea en una frase

> Un número de validación solo vale si responde a la pregunta que de verdad nos importa:
> **¿cómo predecirá el modelo sobre datos que no ha visto?**

Casi toda tu parte (P2.2) consiste en construir y comprobar un procedimiento de validación
que **no haga trampa**, para que el resto de la práctica (P3, P4, P5) mida siempre igual.

Por eso tu rol tiene **veto**: si alguien evalúa de forma que viola el contrato, tú lo paras.

---

## 1. Vocabulario imprescindible

| Término | Significado | Ejemplo en este proyecto |
|---|---|---|
| **Fold** | Una de las K partes en que se divide el dato. En cada vuelta, una parte valida y el resto entrena. | K = 5 folds |
| **Leakage (fuga)** | Información de la parte de validación que "se cuela" en el entrenamiento. | Una réplica de storage en train y su gemela en validación |
| **Grupo** | Conjunto de filas que son "la misma cosa" y no deben separarse. | `config_group` (5 réplicas de una configuración), `feature_group` (mismo vector de características) |
| **Estratificar** | Mantener en cada fold la misma proporción de clases que en el total. | Ransomware ≈ 0,6 % en cada fold |
| **Estimando** | Lo que el número pretende estimar. | "Rendimiento sobre grupos nunca vistos" |
| **Fuera de fold (OOF)** | Predicciones de cada fila hechas por un modelo que **no** la vio entrenando. | `oof_pred` en `evaluate_cyber` |
| **Soporte** | Número de ejemplos reales de una clase. | Ransomware: 208 |
| **K y K_NN** | **K** = nº de folds (5). **K_NN** = nº de vecinos. Nunca se escribe "K" suelto para vecinos. | `n_neighbors=5` es K_NN = 5 |

---

## 2. Las reglas del contrato que más te afectan

1. `row_id`, `feature_group`, `config_group` y `fold_id` **nunca** entran en `.fit()`.
2. Escaladores y encoders se ajustan **solo con el entrenamiento** de cada fold (por eso van dentro de un `Pipeline`).
3. La semilla `2627` se pasa **solo donde hay aleatoriedad real** (por ejemplo `shuffle=True`), nunca a operaciones deterministas, y nunca se "prueba" con otras semillas.
4. Desde que llegan los folds canónicos, **todo resultado puntuado usa los canónicos**. Los nuestros son solo evidencia de que entendemos.
5. Regla log → físico: las métricas en escala log se calculan con `z` y `ẑ`; para nanosegundos se **invierte primero** (`10**ẑ`) y **luego** se calcula el error. **Nunca** `10**MAE(z)` como si fueran ns.

---

## 3. Recorrido sección por sección

### 3.1 Setup, datos y pipelines (secciones 0–2)

**Qué hace.** Carga las 4 tablas, separa predictores del objetivo y define dos `Pipeline`:

- **Cyber:** one-hot en 3 columnas nominales; `log1p` + `StandardScaler` en 8 numéricas; `KNeighborsClassifier(n_neighbors=5)`.
- **Storage:** one-hot en 4 nominales; `StandardScaler` en 5 numéricas; `KNeighborsRegressor(n_neighbors=5)`. El objetivo es `z = log10(latency_ns)`.

**Por qué un `Pipeline`.** Imagina que escalas **todo** el dato antes de validar. La media y la desviación
usadas incluyen filas de validación: has filtrado información del futuro hacia el pasado.
Dentro de un `Pipeline` + `cross_validate`, en cada fold el escalador se ajusta solo con las filas de entrenamiento.

**Por qué `log1p` en cyber.** Columnas como `src_bytes` tienen valores muy desiguales (de 0 a más de 10⁹ bytes).
KNN usa distancias; sin comprimir la escala, una sola columna dominaría la distancia. `log1p` usa `log(1+x)`
porque hay ceros y `log(0)` no existe.

**Por qué escalar.** KNN calcula distancias. Si `block_size` va de 4 a 1.024 y `read_fraction` de 0 a 100, y además
tuvieras una columna que llegara a millones, la de mayor rango dominaría la distancia y las demás apenas contarían.

**Por qué `raid` es categórica.** "4+2" es un tipo de RAID, no una cantidad; si la trataras como número, KNN
creería que "4+2" está "más lejos" de "1" que "0". Por eso va con one-hot.

---

### 3.2 Construir los folds (sección 3, P2.2 a)

```python
cv_cyber = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=PROJECT_RANDOM_STATE)
cv_storage = GroupKFold(n_splits=5, shuffle=True, random_state=PROJECT_RANDOM_STATE)
folds_cyber = list(cv_cyber.split(X_cyber, y_cyber, groups=g_cyber))
```

Cada `fold` es un par `(train_idx, val_idx)` de posiciones de fila.

**Qué hace cada splitter, con un ejemplo mínimo.** Seis filas con grupos A, A, B, B, C, C y 3 folds:

```
fila:   0  1  2  3  4  5
grupo:  A  A  B  B  C  C
```

- `KFold` normal podría dar validación = {0, 2, 4}. Entonces la fila 0 (grupo A) valida mientras su gemela, la fila 1, entrena. **Fuga.**
- `GroupKFold` da validación = {0, 1} (todo A), {2, 3} (todo B), {4, 5} (todo C). Ninguna gemela cruza la frontera.

**`StratifiedGroupKFold`** hace lo mismo pero además intenta que cada fold tenga la misma proporción de clases.
"Intenta" porque la restricción de grupos limita cuánto puede equilibrar.

**Por qué `shuffle=True` y la semilla.** Sin `shuffle`, el reparto es determinista y no consume aleatoriedad,
así que pasar la semilla sería innecesario (y la regla 8 dice que solo se pasa donde hay aleatoriedad).
Con `shuffle=True`, el orden se baraja y la semilla fija el resultado para que sea reproducible.

**Comprobación previa:** `assert (cyber["row_id"] == cyber_groups["row_id"]).all()`. Los grupos se usan **por posición**
(`g_cyber.iloc[...]`); si las tablas estuvieran en distinto orden, estarías agrupando filas equivocadas sin enterarte.

---

### 3.3 Verificar invariantes (sección 4, P2.2 b)

Una "invariante" es algo que debe cumplirse **siempre**. El código lo comprueba con `assert`: si falla, el notebook se detiene.

```python
shared = set(g_cyber.iloc[tr]) & set(g_cyber.iloc[va])   # intersección de grupos
assert not shared
```

Si la intersección de los conjuntos de grupos de train y validación está vacía, ningún grupo cruza la frontera.

**Resultado:**

| | Cyber | Storage |
|---|---|---|
| Grupos compartidos train/val | 0 en los 5 folds | 0 en los 5 folds |
| Clases en cada validación | 10/10 | — |
| Ransomware en validación | 42, 42, 41, 41, 42 | — |
| Configuraciones por validación | — | ≈ 488 |

**Por qué se comprueba que estén las 10 clases.** `StratifiedGroupKFold` no garantiza que una clase pequeña llegue
a todos los folds. Si un fold no tuviera ransomware, su macro F1 para esa clase no se podría calcular.

**Diferencia con CV por filas (lo que hay que saber explicar):** la CV por filas reparte *filas*; la agrupada reparte
*grupos enteros*. La primera puede dejar gemelas a ambos lados; la segunda garantiza que la validación contiene
solo grupos que el modelo no ha visto.

---

### 3.4 Storage: agrupado frente a por filas (sección 5, P2.2 c)

**Experimento controlado:** mismo pipeline, misma métrica, **solo cambia el reparto**.

| Protocolo | MAE(z) medio | Rango entre folds |
|---|---|---|
| `GroupKFold` (agrupado) | **0,212** | 0,019 |
| `KFold` por filas | **0,081** | 0,010 |

El error agrupado es unas **2,6 veces mayor**, y es mayor en los 5 folds (no es casualidad de un fold).

**Mecanismo (explicación completa):**

1. Cada configuración se mide 5 veces. Las 5 réplicas tienen **exactamente las mismas 9 características**;
   solo difieren por ruido de medición en la latencia.
2. Con `KFold` por filas, una fila de validación casi siempre tiene réplicas suyas en entrenamiento.
3. KNN con K_NN = 5 busca los 5 vecinos más cercanos. Las réplicas están a **distancia 0**, así que son sus vecinos.
4. Promedia sus latencias, que son casi la respuesta correcta. El modelo "acierta" por **memoria**, no por generalizar.
5. Con `GroupKFold` las 5 réplicas van juntas, así que los vecinos son de **otras** configuraciones.
   Eso es lo que ocurrirá en el test final oculto.

**Detalle importante: el `shuffle=True` del `KFold`.** Las réplicas están **contiguas** en el fichero
(`S00000`, `S00001`... del mismo grupo). Sin barajar, un `KFold` de 5 partes cortaría el fichero en 5 bloques
consecutivos y casi todas las réplicas quedarían juntas **por casualidad del orden**, escondiendo parte de la fuga.
Esto es un razonamiento mío que **no medí**; si quieres evidencia, se puede ejecutar el `KFold` sin barajar y comparar.

**Cómo expresar el error de z en términos físicos.** 10^0,212 ≈ 1,63 y 10^0,081 ≈ 1,21. Se lee como
**factor multiplicativo**: "las predicciones están típicamente dentro de un factor ≈ 1,6 de la latencia medida".
Nunca como "1,63 ns" ni "0,63 ns".

---

### 3.5 Cyber: agrupado frente a sin agrupar (sección 6, P2.2 d)

| Protocolo | Macro F1 medio | Rango |
|---|---|---|
| `StratifiedGroupKFold` | 0,9442 | 0,012 |
| `StratifiedKFold` | 0,9443 | 0,016 |

Diferencia de medias: **−0,0001**. Es pequeña frente a la dispersión descriptiva de cada protocolo. Los folds con el mismo número no validan las mismas filas bajo los dos protocolos, por lo que no son diferencias pareadas de modelos.

**Por qué casi no hay diferencia aquí.** Hay 34.883 `feature_group` para 34.928 filas: casi todos los grupos son una sola fila.
Hay 38 grupos de más de una fila, que contienen 83 filas en total: son 45 filas adicionales respecto de un representante por grupo. Sus etiquetas están en conflicto. Esto contextualiza la diferencia pequeña, pero no demuestra su causa.

**Lección:** el enunciado dice explícitamente que el signo y tamaño de la diferencia **no se predicen**: se miden.
Y la justificación del agrupado **no depende de cuánto salga**.

---

### 3.6 El argumento metodológico (sección 7, P2.2 e)

Es el corazón conceptual. Estructura:

1. **Qué queremos estimar:** rendimiento sobre un grupo exacto de características **no visto**. Eso es lo que verá el test final.
2. **Qué rompe un grupo partido:** parte de la validación se responde con filas idénticas del entrenamiento
   (distancia 0 en KNN). Se mide *memoria*, no generalización → el score estima **otra cosa**.
3. **El sentido del cambio no está garantizado:** en cyber, la puntuación sin agrupar puede subir, bajar o cambiar poco. El resultado observado no corrige el hecho de que se están evaluando grupos parcialmente vistos.
4. **La selección de modelo hereda el problema:** elegir K_NN con un score contaminado favorece modelos que memorizan.

**Idea clave para la defensa:** el protocolo se elige *por lo que garantiza*, no *por lo que midió*.
Aunque en cyber la diferencia sea 0, el argumento sigue en pie.

---

### 3.7 Folds canónicos (secciones 8 y 9, P2.2 f y g)

El profesor entrega `cyber_folds.csv` y `storage_folds.csv` (`row_id`, `fold_id` de 0 a 4).

**Qué comprobamos:**

1. `row_id` alineado con las tablas de desarrollo.
2. Los invariantes también se cumplen en los canónicos (0 grupos en más de un fold; ransomware 42/42/41/41/42).
3. Comparación con los nuestros mediante una **tabla cruzada** (filas = nuestro fold, columnas = fold canónico).

**Cómo leer la tabla cruzada:**

```
Cyber (idénticos):              Storage (distintos):
canonical fold   0    1    ...  canonical fold   0    1    2   ...
our fold                        our fold
0              6986   0    ...  0              445  510  535  ...
1                 0 6986   ...  1              455  445  510  ...
```

- **Cyber:** cada fila tiene un solo número distinto de cero → los dos repartos son *el mismo* (salvo etiquetas).
- **Storage:** cada fold nuestro se reparte entre los 5 canónicos → son repartos **distintos**, ambos válidos.
  Con `shuffle=True`, `GroupKFold` asigna grupos "al azar", así que otra semilla/versión da otro reparto igual de bueno.
  No sabemos qué semilla usó el curso, y no hace falta.

**Adoptar el canónico:** `PredefinedSplit`.

```python
PredefinedSplit(test_fold=fold_id)   # fold_id[i] = k  → la fila i valida en el fold k
```

Es el adaptador que convierte un vector "a qué fold pertenece cada fila" en un `cv` que acepta `cross_validate`.
La función `canonical_cv("cyber")` lo encapsula para poder copiarlo a otros notebooks sin cambiarlo.

---

### 3.8 Funciones de evaluación comunes (sección 9)

Son las funciones que se reutilizan para evaluar cualquier modelo. Dos ideas centrales:

**A) Clonar y reajustar en cada fold** (`clone(estimator)`): cada fold parte de un modelo **sin entrenar**.
Si reutilizaras el mismo objeto, arrastrarías estado de un fold a otro (fuga entre folds).

**B) Predicciones fuera de fold (OOF).** Cada fila se predice **una sola vez**, con un modelo que no la vio.
Se juntan las 34.928 predicciones y se calcula la tabla por clase y la matriz de confusión sobre ese conjunto.

#### `evaluate_cyber`

Devuelve por fold: macro F1 (**primaria**), weighted F1, accuracy y tiempo; además tabla por clase
(precisión, recall, F1 **junto al soporte**), matriz de confusión con nombres semánticos, y `predict_proba` OOF.

- **Macro F1:** media simple de los F1 de las 10 clases. Cada clase pesa igual, ransomware (0,6 %) pesa lo mismo que ddos (13,7 %).
  Por eso es la métrica primaria en un problema con una clase deliberadamente rara.
- **Weighted F1:** media ponderada por soporte; las clases grandes dominan.
- **Accuracy:** porcentaje de aciertos. Engañosa con clases desbalanceadas.

Ejemplo de por qué importa el soporte: ransomware tiene recall 0,952 sobre 208 filas (198 aciertos).
En cada fold de validación hay ~42, así que **un solo error mueve el recall 1/42 ≈ 2,4 puntos**.
Las métricas de esa clase son muy ruidosas y hay que presentarlas siempre con su soporte.

#### `evaluate_storage`

Devuelve por fold: MAE(z) (**primaria**), RMSE(z), R²(z), y **un** MAE_ns.

- **MAE:** error absoluto medio. Interpretable, robusto a valores extremos.
- **RMSE:** eleva al cuadrado antes de promediar; castiga más los errores grandes.
- **R²:** fracción de la varianza explicada. 0 = no mejora a predecir la media; 1 = perfecto; puede ser negativo.

**La regla log → físico, con un ejemplo numérico.** Supón que el modelo se equivoca *en z* por 0,2 de forma constante.
Eso equivale a un factor 10^0,2 ≈ 1,58 en la latencia, en *cualquier* escala:

| Latencia real | Predicción (×1,58) | Error en ns |
|---|---|---|
| 100.000 ns (10⁵) | 158.000 | 58.000 |
| 2.500.000.000 ns (2,5·10⁹) | 3.950.000.000 | 1.450.000.000 |

El **mismo** error en z da errores en ns 25.000 veces distintos. Por eso:
- `10**MAE(z)` **no** es un error en ns (promedia un factor, no una cantidad).
- El MAE_ns hay que calcularlo **después** de invertir cada predicción (`10**ẑ`) y contra la latencia real,
  y estará dominado por las latencias grandes.

#### `summarise` y otras decisiones

Eliminé una función auxiliar que no se usaba. El tiempo de ajuste (`fit_time_s`) se mide por fold porque la tabla
comparativa lo pide.

---

### 3.9 `comparativa.csv` (sección 10)

Tabla acumulativa: **una fila por (reto, familia, hito, ruta)** y **solo se añade, nunca se reescribe**.

`append_comparativa_row`:
1. Crea el fichero con la cabecera de la plantilla si no existe.
2. Comprueba que las columnas coinciden con la plantilla (`assert`).
3. Si la clave `(milestone, challenge, family, route)` ya existe, **no añade** → re-ejecutar el notebook no duplica filas.
4. Si no, añade la fila al final (`mode="a"`).

**Filas Dummy (baselines) y por qué sus números cuadran** (esto sirve como prueba de que el código es correcto):

- **Cyber `most_frequent`:** predice siempre la misma clase (el 13,74 % de las filas). Accuracy = 0,1374 ✔.
  Esa clase tiene recall 1 y precisión 0,1374 → F1 = 2·0,1374/(1+0,1374) ≈ 0,2416. Las otras 9 clases tienen F1 = 0.
  Macro F1 = 0,2416/10 = **0,0242** ✔.
- **Storage `mean`:** predice siempre la media de z. R²(z) ≈ 0 ✔ (por definición de R²: no mejora a la media).
  MAE(z) = 0,677; MAE_ns ≈ 2,7·10⁷.

Esos valores son el **suelo**: cualquier modelo (KNN, etc.) debe superarlos para justificar su complejidad.
Para comparar, el KNN del protocolo (K_NN = 5) da macro F1 ≈ 0,944 y MAE(z) ≈ 0,21.

---

## 4. La auditoría extra (más allá de lo que pide el enunciado)

Se ejecutó aparte (no está en el notebook). Comprobó, **sin usar la columna de grupos**:

1. Cada fila se valida exactamente una vez; train ∩ val = ∅; train ∪ val = todo.
2. **Hash de cada vector de características**: ningún vector idéntico aparece a ambos lados de un fold
   (0 en cyber, 0 en storage). Esto confirma que los grupos capturan las duplicaciones reales,
   porque no depende de que `feature_group` esté bien construido.
3. Un mismo vector no se reparte entre dos grupos distintos (0 casos).
4. Proporciones de clase por fold iguales a las globales hasta el 2.º decimal.
5. Ransomware: 208 filas = 208 grupos (uno por fila), 41–42 por fold en validación y 166–167 en entrenamiento.

**Hallazgo de datos a tener presente:** 38 `feature_group` contienen **más de una etiqueta** (mismo vector, clases distintas).
Es ambigüedad en los datos (no un fallo de los folds) y limita la precisión máxima alcanzable.

---

## 5. Preguntas probables en la defensa (con respuesta corta)

**¿Por qué grupos también en cyber si la diferencia salió casi nula?**
Porque el protocolo se justifica por lo que el score debe estimar (generalizar a grupos no vistos), no por el tamaño del hueco observado. La escasa duplicación ayuda a contextualizar el resultado; no demuestra por sí sola la causa de la diferencia ni su significación.

**¿Por qué el KFold por filas sale más optimista en storage?**
Porque las 5 réplicas de cada configuración, idénticas en características, quedan repartidas entre train y validación; KNN las encuentra a distancia 0 y "recuerda" la latencia.

**¿Qué es un invariante y cómo lo verificaste?**
Una propiedad que debe cumplirse siempre (ningún grupo en train y validación). Intersección de conjuntos de grupos vacía en cada fold, con `assert`, más la comprobación de que cada grupo corresponde a un vector exacto y de que una configuración no aparece en varios grupos.

**¿Por qué macro F1 como métrica primaria?**
Trata igual a todas las clases; con una clase deliberadamente rara (ransomware, 208 filas) no deja que las grandes la tapen.

**¿Qué se puede decir de ransomware con ~42 ejemplos por fold?**
Que sus métricas son ruidosas (un error mueve el recall ~2,4 puntos) y deben darse junto al soporte; no se pueden hacer afirmaciones finas sobre diferencias pequeñas.

**¿Por qué no `10**MAE(z)` como error en ns?**
Porque un error constante en z es un factor multiplicativo, no una cantidad: el mismo factor da errores en ns enormemente distintos según la latencia. Se invierte la predicción y se calcula el error contra la latencia real.

**¿Por qué el KNN y el escalador van dentro del Pipeline?**
Para que el escalador se ajuste solo con el entrenamiento de cada fold; si no, la validación influiría en la media y desviación usadas.

**¿Por qué tus folds y los canónicos difieren en storage?**
Los dos son agrupados y válidos; difieren en qué configuraciones van a cada fold (asignación aleatoria de grupos con otra semilla o procedimiento). No cambia el protocolo.

**¿La semilla hace el algoritmo determinista?**
No: hace el experimento reproducible *condicionado a esa semilla*. No se puede afirmar que el resultado sea independiente de ella.




## Aclaración de la revisión final

En cyber, una partición sin grupos puede dar una puntuación mayor, menor o similar: no se debe afirmar que siempre la infla. Los 38 grupos repetidos tienen etiquetas en conflicto. Los folds con el mismo número bajo protocolos distintos no validan las mismas filas y no forman una comparación pareada. La comparación pareada de modelos se hace sobre los mismos folds canónicos; su dispersión es descriptiva y no demuestra significación ni equivalencia. Este archivo es una guía interna en español, no uno de los entregables evaluables.
