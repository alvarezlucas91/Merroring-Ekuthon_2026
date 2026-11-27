# Enseñale a la IA un movimiento

Demo del Ekuthon 2026 (Ekumen Technical Showcase). El público escanea un QR, se pone el celular en el antebrazo
como si fuera un smartwatch, inventa un movimiento, lo repite 20 segundos y el modelo lo aprende en el momento,
sin reentrenar. En la pantalla grande se ve un mapa del espacio de embeddings con cada persona moviéndose en vivo
hacia la actividad que está haciendo.

Técnicas que muestra: sensores inerciales de wearables, embeddings, few-shot learning con prototipos,
detección de actividades desconocidas (open-set) y un pipeline de datos en tiempo real.

## Cómo funciona

```
Celular (navegador)          Notebook (servidor)                                  Pantalla grande
acelerómetro ~60 Hz  ──WS──▶ remuestreo a 30 Hz, en g ─▶ ventana de 5 s ─▶ encoder ─▶ vector
paquetes cada 200 ms         compara con prototipos (similitud coseno)                  panel /panel
                             < umbral ─▶ "Desconocida"                    ──WS──▶ mapa 2D + tarjetas
                             grabar 20 s ─▶ promedio de vectores = prototipo nuevo
```

- No se guarda la señal cruda de nadie. Solo se guardan los prototipos (vectores) en `data/prototypes.json`.
- Cada grabación es un prototipo. Si dos personas enseñan el mismo movimiento, la actividad queda con dos
  prototipos y gana el más parecido. Eso mejora mucho el reconocimiento de terceros (ver resultados).

## Instalación (una vez, ~5 minutos)

Requiere Python 3.10 o superior y git.

```bash
pip install -r requirements.txt
python scripts/prepare_wisdm.py          # baja WISDM y arma data/wisdm_watch_30hz.npz (~1 min)
```

Los modelos ya entrenados vienen en `models/`. Para reentrenar o reproducir la evaluación:

```bash
python scripts/train_encoder.py --mode eval     # ~4 min en CPU, sin 6 actividades y 10 personas
python scripts/train_encoder.py --mode deploy   # ~4 min, con todo (lo usa la demo)
python scripts/evaluate_fewshot.py              # genera results/fewshot.png y los umbrales
```

## Correr la demo

```bash
python app/server.py                       # encoder híbrido por defecto
cloudflared tunnel --url http://localhost:8000
```

El túnel imprime una dirección `https://algo.trycloudflare.com`. **Abran el panel con esa dirección**
(`https://algo.trycloudflare.com/panel`): el QR del panel se arma con la dirección desde la que se abrió.
Los celulares necesitan https para leer sensores; por eso el túnel es obligatorio (no alcanza con la IP local).
Alternativa al túnel: `ngrok http 8000`.

Instalar cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

### Sin celulares (prueba o plan B)

```bash
python scripts/simulate_phones.py --phone "Ana=Aplaudir:30,Caminar:30" --phone "Beto=Picar una pelota:60"
```

Reproduce señales reales de WISDM como si fueran celulares conectados.

### Prueba automática de punta a punta

```bash
python scripts/e2e_test.py
```

Tres personas simuladas enseñan 3 movimientos y otras cuatro, distintas, los hacen (más uno que nadie enseñó).
`python scripts/e2e_exercise.py` prueba el modo Ejercicio guiado; con `--keep` deja todo andando en
http://localhost:8000/panel para mirarlo.

## Guion sugerido (5 minutos)

1. Mostrar el QR. Dos o tres personas se conectan y se ponen el celular en el antebrazo.
2. Una persona inventa un movimiento repetitivo y bien marcado (picar una pelota imaginaria, escribir en el aire, revolver una olla).
   En el panel: escribir el nombre en su tarjeta y tocar **Enseñar**. 3 segundos de cuenta regresiva y 20 de grabación.
3. Aparece la nube nueva en el mapa. La persona repite el movimiento y su círculo viaja a esa nube.
4. Otra persona hace algo que nadie enseñó: el sistema dice **Desconocida**. Es el momento para explicar open-set.
5. Una segunda persona enseña el mismo movimiento: ahora lo reconoce mejor en terceros.
6. Cerrar con `results/fewshot.png`: el resultado medido con 10 personas y 6 actividades que el modelo nunca vio.

## Resultados (evaluación offline)

![few-shot](results/fewshot.png)

Medido con WISDM 2019 (reloj en la muñeca), sobre 6 actividades y 10 personas que el encoder propio nunca vio
al entrenar. Precisión eligiendo entre las 6 actividades nuevas:

| Encoder | Misma persona, 20 s | Enseña una, reconoce a otra, 20 s | Enseñan dos, reconoce a una tercera, 20 s | Detección de desconocido (AUROC) |
|---|---|---|---|---|
| **Híbrido: features + CNN propia** (default) | **94%** | **72%** | **80%** | **0.97** |
| Features clásicas | 94% | 71% | 76% | 0.94 |
| CNN propia (Ekumen) | 91% | 69% | 72% | 0.94 |
| Oxford SSL, harnet5 (solo uso académico) | 78% | 58% | 66% | 0.80 |

Lectura honesta:
- Para la misma persona funciona muy bien desde 5 segundos de grabación (91%).
- Entre personas distintas baja, porque cada uno hace el mismo movimiento a su manera. Por eso en la demo conviene
  que lo enseñen dos personas. Entre personas distintas (20 s de grabación), por actividad: escribir a mano 98%,
  picar una pelota 92%, doblar ropa 69%, patear 62%, aplaudir 55%, cepillarse los dientes 54%.
  Los movimientos que todos hacen parecido funcionan mucho mejor que los que cada uno hace a su manera.
- El modelo autosupervisado de Oxford, usado como extractor sin ajuste fino, quedó por debajo. El híbrido
  (features clásicas + CNN chica propia) ganó en todos los escenarios.

Detalle numérico: `results/fewshot_metrics.json`.

## Antes del evento (checklist)

- Brazaletes de running para celular (uno por persona en escena) y un celular propio de respaldo.
- Probar con un iPhone y un Android reales: permisos, calibración y que la pantalla no se apague.
- Ensayar con gente que no hizo el proyecto.
- Ajustar el umbral de "desconocida" en **Controles** si en vivo dice "Desconocida" demasiado (bajarlo)
      o casi nunca (subirlo). El valor de partida sale de la evaluación (0.64 para el híbrido).
- Tener el simulador listo como plan B por si falla la red.
- Tocar **Borrar todo** antes de empezar.

## Detalles técnicos

- **Ventanas:** 5 s a 30 Hz (150 muestras × 3 ejes), en g, gravedad incluida. Se clasifica cada 1 s y se vota
  entre las últimas 3 decisiones para que no parpadee.
- **Calibración del celular:** en iPhone y Android el signo del acelerómetro puede venir invertido. La página
  pide apoyar el celular boca arriba 2 s y fija la convención "boca arriba = +1 g en z", la misma del dataset.
- **Encoder híbrido:** 67 features clásicas (estadísticas por eje y magnitud, espectro, correlaciones) estandarizadas
  + embedding de 128 de una CNN 1D propia (4 bloques convolucionales, clasificador coseno, aumentación con rotaciones
  de hasta 30°, ruido y escala), concatenados y normalizados.
- **Desconocido:** si la similitud con el prototipo más parecido es menor al umbral, se responde "Desconocida".
- **Mapa:** PCA 2D sobre los vectores de las grabaciones. Se recalcula cada vez que se aprende algo.
- **Actividades base (experimental):** el botón en Controles precarga 5 actividades desde WISDM. Esos datos son de
  un reloj en la muñeca, no de un celular en el antebrazo, así que pueden fallar en vivo.
- `--encoder features|ekumen|hybrid|oxford` en `app/server.py` cambia el encoder.

## Estructura

```
app/
  server.py        FastAPI + WebSockets (celular y panel)
  engine.py        ventanas, embeddings, prototipos, grabación, mapa 2D
  exercise.py      modo Ejercicio guiado: reposo, repeticiones, fases, DTW, puntajes
  encoders.py      features clásicas, CNN propia, híbrido, Oxford
  dsp.py           remuestreo, ventanas, features
  static/phone.html, static/panel.html
scripts/
  prepare_wisdm.py, train_encoder.py, evaluate_fewshot.py, simulate_phones.py, e2e_test.py
  simulate_exercise.py, e2e_exercise.py   (movimientos sintéticos y prueba del modo ejercicio)
models/            encoder entrenado, normalizaciones y umbrales
results/           gráfico y métricas de la evaluación
```

## Modo "Ejercicio guiado" (imitar un movimiento)

Pestaña **Ejercicio guiado** en el panel. Se graban movimientos de referencia (los que quieras; quedan guardados
aunque se reinicie el servidor) y cada persona elige cuál imitar, con tempo o sin tempo.

**Grabar un movimiento**
1. En la tarjeta de quien enseña: **Grabar referencia**.
2. La persona se queda quieta unos 2 segundos (el celular toma su posición de reposo) y hace el movimiento.
   Puede tener pausas de cualquier duración y terminar en cualquier posición.
3. Al terminar, **Terminar**. Aparece el cartel **Guardar movimiento** con la curva, las fases detectadas,
   el nombre del ejercicio y los nombres de las fases (opcional). **Guardar** o **Descartar**.
4. Los movimientos guardados se apilan a la derecha (el más nuevo arriba). Tocando uno se ve su curva y su ranking;
   "borrar" (dos toques) lo elimina.

La quietud entre el final del movimiento y el toque en Terminar se recorta: no cuenta en la duración.

**Imitar**
1. En la tarjeta de otra persona: **Imitar…**. En el cartel se elige el movimiento y el modo:
   - **Con tempo:** cuenta cuánto tarda cada fase, la amplitud y la forma (40% / 30% / 30%).
   - **Sin tempo:** solo cuánto se parece el movimiento, sin importar la velocidad (60% forma / 40% amplitud).
     Los tiempos se muestran, pero no suman.
2. La persona se queda quieta 2 segundos y repite el movimiento las veces que quiera. Cada repetición se cierra sola:
   - cuando vuelve a la posición inicial y queda quieta 0,35 s, si ya hizo tantos tramos de movimiento como
     la referencia (así una pausa en la posición inicial a mitad del ejercicio no corta la repetición), o
   - cuando queda quieta en cualquier posición más de 1,5 veces la pausa más larga de la referencia
     (mínimo 2,5 s; si la referencia tiene una pausa de 5 s, el corte es a los 7,5 s), o
   - con el botón **Cerrar repetición**.
3. **Terminar** cierra la sesión de esa persona. El ranking queda guardado por movimiento y por modo.

**Dónde va el celular:** sujeto a la parte del cuerpo que se mueve. Para la mano o el brazo, en el antebrazo;
para una sentadilla, en el muslo o en el bolsillo delantero. Instructor e imitador en el mismo lugar del cuerpo.
La orientación del celular puede ser distinta entre personas: la comparación no depende de eso.

**Qué mide**
- **Señal:** ángulo que rota el celular respecto de la posición de reposo (con la gravedad), velocidad de giro
  (giroscopio; si el celular no lo informa, se estima) e intensidad de la aceleración sin gravedad.
- **Inicio y fin exactos:** donde la velocidad de giro cruza el 10% de su pico (criterio habitual en análisis de movimiento).
- **Fases:** cada pausa separa fases. Sin pausas, el movimiento se divide en ida y vuelta en el punto de máximo ángulo.
- **Comparación:** DTW (Dynamic Time Warping) alinea la curva del intento con la de referencia. Con ese alineamiento
  se ubica cada fase en el intento y se compara su duración.

**Validación con movimientos sintéticos** (`python scripts/e2e_exercise.py`). Cada persona con el celular orientado distinto:

| Movimiento | Imitación | Modo | Total | Tiempo | Amplitud | Forma |
|---|---|---|---|---|---|---|
| Subir la mano (2 / 1 / 2 s, 90°) | igual | con tempo | 99% | 96% | 100% | 100% |
| Subir la mano | al doble de velocidad | con tempo | 66% | 23% | 92% | 97% |
| Subir la mano | al doble de velocidad | sin tempo | 95% | (no suma) | 92% | 97% |
| Con pausas de 5 s | pausas de 6 y 4 s | con tempo | 95% | 88% | 100% | 100% |
| Termina arriba (no vuelve al inicio) | igual | con tempo | 97% | 94% | 100% | 100% |

Un perfil de 2/1/2 s se mide como 1,9/1,1/1,9 s y uno de 4/2/3 s como 3,6/2,3/2,8 s: el arranque y el frenado
suaves quedan fuera de las fases. El efecto es igual en la referencia y en la imitación.

**Límites:** funciona mejor con movimientos que rotan la parte del cuerpo donde está el celular. Un desplazamiento
sin rotación (deslizar la mano en línea recta sin girarla) da una señal débil. Si alguien encadena repeticiones sin
ninguna pausa entre ellas, pueden unirse en una sola: conviene una pausa breve al volver al inicio.

**Ajustes en vivo (Controles):** "Quieto por debajo de" (subirlo si a alguien le cuesta que el sistema detecte que
está quieto), "Piso del umbral de pausa" (subirlo si las pausas no se detectan porque la persona tiembla al sostener)
y "Exigencia en tiempo". Los movimientos se guardan en `data/exercises.json`.

## Licencias y créditos

- **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset**, Gary Weiss (2019), UCI Machine Learning
  Repository, https://doi.org/10.24432/C5HK59, CC BY 4.0. `prepare_wisdm.py` intenta usar un espejo en GitHub de los
  archivos crudos; la fuente oficial es https://archive.ics.uci.edu/dataset/507.
- **Encoder propio y features:** código de este proyecto, entrenado solo con WISDM. Se puede usar en material comercial
  respetando la atribución de WISDM.
- **Oxford ssl-wearables (opcional):** Copyright © 2022, University of Oxford. Licencia **solo académica, no comercial**.
  No está incluido. Para la comparación: `git clone https://github.com/OxWearables/ssl-wearables vendor/ssl-wearables`
  y `python app/server.py --encoder oxford`. No usarlo en material para clientes sin revisión legal.
