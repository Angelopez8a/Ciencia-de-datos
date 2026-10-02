# Apuntes interactivos · Diplomado en Ciencia de Datos

Guía de estudio en HTML de los 4 módulos del diplomado (UNAM · FES Acatlán), hecha a partir de las
presentaciones, documentos y notebooks del curso. Cada tema incluye explicación, fórmulas, diagramas,
gráficas, simuladores interactivos y código Python **con la salida real que produce**. Los ejemplos usan
datos reales de fuentes públicas y documentadas (ver `codigo/datos/README.md`). El diseño es de estilo
neumórfico, con modo claro y oscuro.

| Página | Contenido |
|---|---|
| `index.html` | Portada y mapa del diplomado |
| `modulo1.html` | Conceptos base, proceso de ciencia de datos, herramientas, TAD, ventanas de tiempo, pandas desde lo básico, integración, ingeniería de variables, limpieza, VarClus, estandarización (fit/transform), PCA, t-SNE, dummies, credit scoring (discretización, WoE, IV, scorecard), interpretación de la logística, OLTP/OLAP |
| `examen-m1.html` | Guía de conceptos clave (fórmulas, decisiones, concepto → código), 15 ejercicios resueltos, 78 preguntas con explicación y 39 tarjetas de repaso |
| `modulo2.html` | Regresión lineal y supuestos, logística, LASSO/Ridge/ElasticNet, gradiente, LDA, kernel, SVM, KNN, Bayes ingenuo, árboles, redes, ensambles, métricas, ROC, KS, Lift, PSI |
| `modulo3.html` | PCA paso a paso, K-means, clustering jerárquico, DBSCAN, ciclo de clustering |
| `modulo4.html` | Deep learning: MLP, activaciones, backprop, CNN, dropout, optimizadores, RNN, embeddings, LSTM/GRU, transfer learning |
| `examen-m4.html` | Hoja de fórmulas, 14 ejercicios resueltos, 70 preguntas con explicación y 32 tarjetas de repaso |

## Estructura

```
sitio-web/
├── index.html, modulo1..4.html, examen-m1.html, examen-m4.html
├── assets/
│   ├── css/style.css         estilos (modo claro y oscuro)
│   └── js/                   main.js (común), m1..m4.js (gráficas y simuladores), quiz-m1.js, quiz.js (M4), data.js
├── codigo/
│   ├── requirements.txt
│   ├── datos/                conjuntos de datos reales copiados sin modificar, con sus fuentes (README.md)
│   └── modulo1..4/*.py       48 programas; su salida es la que aparece en las páginas
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

- Los programas leen sus datos de `codigo/datos/`; si un archivo no está (por ejemplo, al descargar un solo
  `.py`), lo descargan de su fuente original, que aparece en el propio código.
- Los programas `m4_k*.py` usan TensorFlow/Keras. La primera vez, Keras descarga los datos de IMDB y MNIST y
  `m4_k4_transfer_learning.py` descarga los pesos de VGG16 (~58 MB; con `PESOS = None` se obtienen los mismos
  conteos de parámetros sin internet).
- Los ejemplos que entrenan redes usan semillas fijas y operaciones deterministas; en otro equipo o con otra
  versión de TensorFlow los resultados pueden variar ligeramente.
- Verificado con Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9, statsmodels 0.15 y
  TensorFlow 2.21.

## Notas

- El Módulo 3 no tenía presentaciones en la carpeta original (solo la lista de prácticas), así que su contenido
  se desarrolló a partir de esos temas: reducción de dimensiones, clustering jerárquico, DBSCAN y ciclo de clustering.
- Cuando un ejemplo usa datos sintéticos o una tabla ilustrativa del curso (por ejemplo, las «lunas» de DBSCAN o la
  tabla de autos de Bayes ingenuo), la página lo indica de forma explícita.
- Material de estudio personal elaborado a partir de los apuntes del curso.
