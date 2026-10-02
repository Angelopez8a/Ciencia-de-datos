# Apuntes interactivos · Diplomado en Ciencia de Datos

Guía de estudio en HTML de los 4 módulos del diplomado (UNAM · FES Acatlán), hecha a partir de las
presentaciones, documentos y notebooks del curso. Cada tema incluye explicación, fórmulas, diagramas,
gráficas, simuladores interactivos y código Python **con la salida real que produce**.

| Página | Contenido |
|---|---|
| `index.html` | Portada y mapa del diplomado |
| `modulo1.html` | Proceso de ciencia de datos, TAD, ventanas de tiempo, limpieza, VarClus, estandarización, PCA, t-SNE, dummies, credit scoring (WoE, IV, scorecard), OLTP/OLAP |
| `modulo2.html` | Regresión lineal y supuestos, logística, LASSO/Ridge/ElasticNet, gradiente, LDA, kernel, SVM, KNN, Bayes ingenuo, árboles, redes, ensambles, métricas, ROC, KS, Lift, PSI |
| `modulo3.html` | PCA paso a paso, K-means, clustering jerárquico, DBSCAN, ciclo de clustering |
| `modulo4.html` | Deep learning: MLP, activaciones, backprop, CNN, dropout, optimizadores, RNN, embeddings, LSTM/GRU, transfer learning |
| `examen-m4.html` | Hoja de fórmulas, 14 ejercicios resueltos, 70 preguntas con explicación y 32 tarjetas de repaso |

## Estructura

```
sitio-web/
├── index.html, modulo1..4.html, examen-m4.html
├── assets/
│   ├── css/style.css         estilos (modo claro y oscuro)
│   └── js/                   main.js (común), m1..m4.js (gráficas y simuladores), quiz.js, data.js
├── codigo/
│   ├── requirements.txt
│   └── modulo1..4/*.py       40 programas; su salida es la que aparece en las páginas
└── .nojekyll                 le indica a GitHub Pages que publique los archivos tal cual
```

Las librerías externas (Chart.js, KaTeX y highlight.js) se cargan desde cdnjs, así que no hay que instalar nada
para ver el sitio.

## Verlo en tu computadora

Abre una terminal en esta carpeta y ejecuta:

```bash
python3 -m http.server 8000
```

Luego entra a <http://localhost:8000>. (Abrir los `.html` con doble clic también funciona, pero el servidor
local evita restricciones del navegador con archivos locales.)

## Publicarlo en GitHub Pages

1. Crea un repositorio nuevo en GitHub (por ejemplo `apuntes-diplomado`), público y vacío.
2. Desde esta carpeta (`sitio-web`):

   ```bash
   git init
   git add .
   git commit -m "Apuntes del diplomado de ciencia de datos"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/apuntes-diplomado.git
   git push -u origin main
   ```

3. En GitHub: **Settings → Pages → Build and deployment → Source: "Deploy from a branch"**, elige
   la rama `main` y la carpeta `/ (root)`, y guarda.
4. En 1–2 minutos el sitio queda en `https://TU_USUARIO.github.io/apuntes-diplomado/`.

Cada vez que cambies algo: `git add . && git commit -m "cambio" && git push` y GitHub Pages se actualiza solo.

## Ejecutar los ejemplos de código

```bash
pip install -r codigo/requirements.txt
python codigo/modulo4/m4_04_convolucion_pooling.py
```

- Los programas `m4_k*.py` usan TensorFlow/Keras; `m4_k4_transfer_learning.py` descarga los pesos de VGG16
  (~58 MB) la primera vez (o usa `PESOS = None` para probar sin internet; los conteos de parámetros son los mismos).
- Los ejemplos que entrenan redes usan semillas fijas, pero el accuracy puede variar ligeramente entre equipos.
- Verificado con Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9 y TensorFlow 2.21.

## Notas

- El Módulo 3 no tenía presentaciones en la carpeta original (solo la lista de prácticas), así que su contenido
  se desarrolló a partir de esos temas: reducción de dimensiones, clustering jerárquico, DBSCAN y ciclo de clustering.
- Material de estudio personal elaborado a partir de los apuntes del curso.
