/* =========================================================
   quiz.js · cuestionario y tarjetas de repaso del Módulo 4
   Cada pregunta: [tema, pregunta, [opciones], índice correcto, explicación]
   ========================================================= */
(function () {
  "use strict";
  const Q = [
    // ---------- Redes neuronales ----------
    ["Redes neuronales", "El perceptrón multicapa (MLP) más simple tiene…", ["2 capas: entrada y salida", "3 capas: entrada, oculta y salida", "5 capas ocultas", "una sola capa con softmax"], 1, "Entrada (una neurona por variable, sin pesos), oculta (transformación sigmoidal) y salida."],
    ["Redes neuronales", "La capa de entrada de un MLP…", ["aplica una sigmoide a cada variable", "tiene una neurona por variable y NO tiene pesos", "tiene tantas neuronas como clases", "aplica softmax"], 1, "Solo pasa los valores de las variables a la capa oculta."],
    ["Redes neuronales", "Según el curso, una red neuronal se considera profunda a partir de…", ["2 capas ocultas", "3 capas ocultas", "5 capas ocultas", "100 neuronas"], 2, "Con menos de 5 capas ocultas se habla de una red superficial (shallow)."],
    ["Redes neuronales", "¿Más neuronas significa mejor desempeño?", ["Sí, siempre", "No necesariamente: más parámetros traen sobreajuste, más cómputo y problemas de gradiente", "Solo en regresión", "Solo si se usa ReLU"], 1, "La disrupción del DL no está en el número de capas sino en el tipo de capas (p. ej. convoluciones)."],
    ["Redes neuronales", "¿Por qué los pesos se inicializan con valores aleatorios?", ["Para que el entrenamiento sea más lento", "Para que cada neurona se oriente a resultados distintos (romper la simetría)", "Porque Keras lo exige", "Para evitar usar learning rate"], 1, "Si todas empezaran igual recibirían el mismo gradiente y aprenderían lo mismo."],
    ["Redes neuronales", "Una época (epoch) es…", ["una actualización de pesos con un lote", "una pasada hacia adelante y hacia atrás de TODOS los datos (todos los lotes)", "el número de capas ocultas", "el tamaño del lote"], 1, "Una iteración = un lote; una época = todos los lotes una vez."],
    ["Redes neuronales", "Con 1,000 ejemplos y batch size de 50, ¿cuántas iteraciones hay por época?", ["50", "20", "1,000", "1"], 1, "1000 / 50 = 20 lotes por época."],
    ["Redes neuronales", "Backpropagation consiste en…", ["mover la entrada hacia la salida", "que el error y su gradiente fluyan de la capa de salida hacia las ocultas para actualizar los pesos", "eliminar neuronas al azar", "normalizar los datos"], 1, "Usa la regla de la cadena para obtener la derivada de la pérdida respecto a cada peso."],
    ["Redes neuronales", "Para clasificar en K clases excluyentes (una por ejemplo), la salida y pérdida adecuadas son…", ["1 neurona lineal + MSE", "K neuronas softmax + categorical cross-entropy", "1 neurona sigmoide + MSE", "K neuronas ReLU + hinge"], 1, "Softmax da K probabilidades que suman 1; la entropía cruzada mide el error."],
    ["Redes neuronales", "En el argot del curso, una neurona es…", ["un estimador débil", "un ensamble", "una capa completa", "un optimizador"], 0, "Muchas neuronas (estimadores débiles) juntas forman algo parecido a un ensamble."],
    ["Redes neuronales", "Una red de 2 entradas → 3 ocultas → 1 salida (todas densas) tiene… parámetros", ["6", "9", "13", "12"], 2, "Oculta: 2·3+3 = 9. Salida: 3·1+1 = 4. Total 13."],
    // ---------- Activaciones ----------
    ["Activaciones", "El rango de la función sigmoide es…", ["(−1, 1)", "(0, 1)", "[0, ∞)", "(−∞, ∞)"], 1, "Es la función logística, la misma de la regresión logística."],
    ["Activaciones", "El rango de tanh es…", ["(−1, 1)", "(0, 1)", "[0, ∞)", "(0, 0.25)"], 0, "Misma forma de S que la sigmoide pero centrada en 0."],
    ["Activaciones", "ReLU(−3) vale…", ["−3", "0", "3", "0.047"], 1, "ReLU(x) = max(0, x)."],
    ["Activaciones", "¿Por qué ReLU es la más usada en capas ocultas?", ["Porque su salida suma 1", "Es simple y menos susceptible al desvanecimiento del gradiente (derivada constante)", "Porque está acotada en (0,1)", "Porque no tiene derivada"], 1, "Su derivada es 1 para x>0, lo que agiliza el entrenamiento."],
    ["Activaciones", "Problema característico de ReLU:", ["Saturación en ambos extremos", "Unidades 'muertas' que siempre dan 0", "Que suma 1", "Que es lineal"], 1, "Si una neurona solo recibe entradas negativas, su gradiente es 0 y deja de aprender."],
    ["Activaciones", "Si cada imagen puede contener varias frutas a la vez, en la salida conviene…", ["softmax", "varias regresiones logísticas (una sigmoide por clase)", "ReLU", "tanh"], 1, "Softmax asume que cada ejemplo pertenece exactamente a una clase."],
    ["Activaciones", "A una activación cuyo rango es limitado también se le llama…", ["función de pérdida", "función de aplastamiento (squashing)", "kernel", "optimizador"], 1, "También se le llama función de transferencia."],
    ["Activaciones", "La derivada máxima de la sigmoide es…", ["1", "0.5", "0.25", "0"], 2, "σ'(0) = 0.5·0.5 = 0.25: por eso multiplicarla muchas veces desvanece el gradiente."],
    ["Activaciones", "softmax([2, 1, 0.1]) ≈", ["[0.5, 0.3, 0.2]", "[0.659, 0.242, 0.099]", "[0.88, 0.73, 0.52]", "[2, 1, 0.1]"], 1, "e²=7.389, e¹=2.718, e^0.1=1.105; suma 11.212."],
    // ---------- Feedforward ----------
    ["Feedforward", "Una red feedforward se caracteriza porque…", ["sus conexiones forman ciclos", "sus conexiones NO forman ciclos: la información va en una sola dirección", "solo tiene una neurona", "usa memoria del pasado"], 1, "Lo opuesto es una red recurrente."],
    ["Feedforward", "El perceptrón de una sola capa produce…", ["una probabilidad entre 0 y 1", "1 si la suma ponderada supera el umbral (0) y −1 en otro caso", "un vector softmax", "el gradiente"], 1, "Es un clasificador lineal con umbral."],
    ["Feedforward", "La regla delta…", ["apaga neuronas al azar", "compara la salida con el valor esperado y ajusta los pesos en proporción al error", "reduce el tamaño de la imagen", "acumula gradientes al cuadrado"], 1, "w ← w + η(y − ŷ)x; en un MLP el proceso análogo es la retropropagación."],
    // ---------- CNN ----------
    ["CNN", "Un kernel (filtro) en una CNN es…", ["la imagen completa", "una matriz pequeña (3×3, 5×5, 7×7) que se desliza sobre la imagen para extraer características", "una función de pérdida", "una capa densa"], 1, "Sus valores se aprenden durante el entrenamiento."],
    ["CNN", "Entrada 28×28, kernel 3×3, sin padding, stride 1. Tamaño de salida:", ["28×28", "26×26", "25×25", "14×14"], 1, "O = (28 − 3 + 0)/1 + 1 = 26."],
    ["CNN", "Para que la salida conserve el tamaño de la entrada se usa…", ["stride 2", "padding (relleno de ceros, 'same')", "max pooling", "dropout"], 1, "Con kernel 3×3, padding 1 mantiene el tamaño."],
    ["CNN", "Max pooling 2×2 con stride 2 sobre un volumen 8×8×16 produce…", ["8×8×16", "4×4×16", "4×4×8", "16×16×16"], 1, "Reduce alto y ancho a la mitad; la profundidad no cambia."],
    ["CNN", "Parámetros de una Conv2D con 32 filtros 3×3 sobre una imagen RGB:", ["288", "320", "896", "9,248"], 2, "(3·3·3 + 1)·32 = 28·32 = 896."],
    ["CNN", "El flattening…", ["reduce la imagen a la mitad", "convierte la matriz/volumen en un vector para alimentar las capas densas", "aplica ReLU", "rellena con ceros"], 1, "Se toman los números fila por fila y se ponen en una sola columna."],
    ["CNN", "¿Cuál es una VENTAJA de las CNN según la clase?", ["Ejemplos adversarios", "Peso compartido e invarianza traslacional", "Marco de coordenadas", "Lentitud por maxpool"], 1, "Las otras opciones son desventajas."],
    ["CNN", "Una DESVENTAJA de las CNN es…", ["detectan características sin supervisión humana", "los ejemplos adversarios: con un poco de ruido la red ve otra cosa", "son computacionalmente eficientes", "comparten pesos"], 1, "Un humano reconoce la misma imagen con ruido; la CNN puede no hacerlo."],
    ["CNN", "Forma (Keras) de un lote de 32 imágenes RGB de 224×224:", ["(224, 224, 3, 32)", "(32, 224, 224, 3)", "(3, 224, 224)", "(32, 3)"], 1, "(N, alto, ancho, canales)."],
    ["CNN", "La profundidad del volumen de salida de una capa convolucional es igual a…", ["el tamaño del kernel", "el número de filtros", "el stride", "el número de canales de entrada"], 1, "Cada filtro produce un mapa de activación; se apilan en profundidad."],
    ["CNN", "En la convolución RGB de la clase los canales dan 308, −498 y 164 con sesgo 1. La salida es…", ["−26", "−25", "971", "−24"], 1, "308 − 498 + 164 + 1 = −25."],
    // ---------- Dropout y arquitecturas ----------
    ["Dropout y arquitecturas", "La tasa de abandono típica de dropout es…", ["0.05", "0.5", "0.95", "1"], 1, "Corresponde a apagar ~50 % de las neuronas en cada iteración."],
    ["Dropout y arquitecturas", "Durante la predicción, dropout…", ["apaga el 50 % de las neuronas", "se desactiva: se usan todas las neuronas", "duplica las neuronas", "elimina capas"], 1, "Solo actúa durante el entrenamiento."],
    ["Dropout y arquitecturas", "Dropout puede interpretarse como…", ["un optimizador", "una técnica de ensamble de muchas subredes", "un tipo de pooling", "una función de pérdida"], 1, "Cada iteración entrena una combinación distinta de neuronas."],
    ["Dropout y arquitecturas", "AlexNet tiene…", ["2 conv + 3 FC = 5 capas", "5 conv + 3 FC = 8 capas", "13 conv + 3 FC = 16 capas", "50 capas"], 1, "LeNet-5 = 5, AlexNet = 8, VGG16 = 16."],
    ["Dropout y arquitecturas", "Una arquitectura 'más ancha' (wider) significa…", ["más capas", "más filtros (mapas de características) por capa", "imágenes de mayor resolución", "menos parámetros"], 1, "Más profunda = más capas; mayor resolución = imágenes más grandes."],
    // ---------- Optimizadores ----------
    ["Optimizadores", "Un learning rate demasiado alto provoca…", ["que el modelo tarde mucho pero llegue", "actualizaciones drásticas que divergen", "que el gradiente sea 0", "más parámetros"], 1, "Uno muy bajo requiere muchísimos pasos; el adecuado llega rápido."],
    ["Optimizadores", "SGD con momentum…", ["usa la magnitud del cambio en lugar de learning rate", "simula la inercia: retiene parte de la dirección anterior y puede escapar de mínimos locales", "acumula gradientes al cuadrado", "apaga neuronas"], 1, "v ← βv + ∇J; θ ← θ − αv."],
    ["Optimizadores", "Optimizador descrito como 'sin tasa de aprendizaje', propuesto por Zeiler (2012):", ["Adam", "RMSProp", "Adadelta", "Adagrad"], 2, "Usa la magnitud de los cambios pasados como calibración."],
    ["Optimizadores", "¿Qué problema de Adagrad resuelve RMSProp?", ["Que no tiene momentum", "Que el learning rate se vuelve demasiado pequeño al acumular todos los gradientes²", "Que no usa derivadas", "Que necesita GPU"], 1, "RMSProp usa una media móvil exponencial de gradientes² recientes."],
    ["Optimizadores", "Adam combina…", ["SGD y dropout", "momentum (promedio de gradientes) y RMSProp (promedio de gradientes²)", "Adagrad y pooling", "LSTM y GRU"], 1, "Por eso combina las ventajas de ambos métodos."],
    ["Optimizadores", "Adagrad es especialmente útil cuando…", ["los datos son dispersos o los parámetros tienen escalas distintas", "hay una sola variable", "no hay gradiente", "se usa dropout"], 0, "Adapta el ritmo de aprendizaje de cada parámetro según sus gradientes históricos."],
    // ---------- RNN ----------
    ["RNN", "Lo que distingue a una RNN es…", ["su capa de pooling", "su 'memoria': la salida depende de entradas anteriores", "que no tiene pesos", "que solo procesa imágenes"], 1, "Toma información de entradas previas para influir en la salida actual."],
    ["RNN", "BPTT se diferencia de backprop tradicional en que…", ["no usa gradientes", "suma los errores de cada paso de tiempo porque los pesos se comparten", "solo actualiza la última capa", "usa softmax"], 1, "Las feedforward no necesitan sumar errores."],
    ["RNN", "En el autocompletado de letras (inglés), la salida softmax tiene tamaño…", ["2", "10", "26", "128"], 2, "Una probabilidad por letra; la entrada es un one-hot de tamaño 26."],
    ["RNN", "Clasificar el sentimiento de una reseña es una RNN…", ["uno a uno", "uno a muchos", "muchos a uno", "muchos a muchos"], 2, "Entra una secuencia de palabras y sale una etiqueta."],
    ["RNN", "La traducción automática es una RNN…", ["uno a uno", "uno a muchos", "muchos a uno", "muchos a muchos (longitudes distintas)"], 3, "Arquitectura codificador–decodificador."],
    ["RNN", "Generar música a partir de una nota inicial es…", ["uno a muchos", "muchos a uno", "uno a uno", "recursiva"], 0, "Una entrada genera una secuencia."],
    ["RNN", "Una RNN bidireccional (BRNN)…", ["usa también información futura de la secuencia", "tiene dos capas de pooling", "no tiene memoria", "solo funciona con imágenes"], 0, "Ej.: adivinar 'under' en 'feeling under the weather' conociendo 'weather'."],
    ["RNN", "Los gradientes explosivos provocan que…", ["el modelo deje de aprender porque el gradiente ≈ 0", "los pesos crezcan demasiado hasta volverse NaN (modelo inestable)", "se apaguen neuronas", "baje el learning rate"], 1, "El caso ≈0 es el gradiente que desaparece."],
    ["RNN", "Solución al vanishing/exploding gradient que menciona la diapositiva:", ["agregar más capas ocultas", "reducir la cantidad de capas ocultas", "quitar las activaciones", "usar softmax"], 1, "En la práctica también: LSTM/GRU, gradient clipping, ReLU."],
    // ---------- Embeddings ----------
    ["Embeddings", "La similitud del coseno entre los one-hot de 'good' y 'great' es…", ["1", "0", "−1", "0.5"], 1, "Son ortogonales: el one-hot no captura significado."],
    ["Embeddings", "Según la clase, la dimensión de un embedding suele ir de…", ["1 a 2", "8 (datos pequeños) a 1024 (datos grandes)", "10,000 a 1,000,000", "siempre 300"], 1, "Es un hiperparámetro; más dimensiones requieren más datos."],
    ["Embeddings", "Los valores de un word embedding…", ["se escriben a mano", "son parámetros entrenables que aprende el modelo", "son siempre 0 o 1", "son la frecuencia de la palabra"], 1, "Se inicializan aleatorios y se ajustan como los pesos de una capa densa."],
    ["Embeddings", "Si las palabras están codificadas con enteros de 0 a 10, input_dim es…", ["10", "11", "1", "100"], 1, "Tamaño del vocabulario = 11 valores distintos."],
    ["Embeddings", "Parámetros de Embedding(input_dim=5000, output_dim=32):", ["5,032", "160,000", "37", "16,000"], 1, "5000 · 32 = 160,000."],
    ["Embeddings", "Cargar embeddings preentrenados (word2vec, GloVe) en la capa Embedding es un ejemplo de…", ["dropout", "transfer learning", "pooling", "BPTT"], 1, "Es uno de los tres usos de la capa según la clase."],
    // ---------- LSTM / GRU ----------
    ["LSTM/GRU", "La LSTM fue introducida por…", ["Hinton y LeCun", "Hochreiter y Schmidhuber", "Freund y Schapire", "Mikolov"], 1, "Como solución al problema del vanishing gradient."],
    ["LSTM/GRU", "Las puertas de una GRU son…", ["olvido, entrada y salida", "reinicio (reset) y actualización (update)", "solo salida", "convolución y pooling"], 1, "La LSTM tiene olvido, entrada y salida."],
    ["LSTM/GRU", "En una GRU, el nuevo estado es…", ["c_t = G_u·c̃_t + G_f·c_{t−1}", "c_t = G_u·c̃_t + (1 − G_u)·c_{t−1}", "c_t = tanh(x_t)", "c_t = c_{t−1}"], 1, "Solo se escribe en la parte que se borró; además a_t = c_t."],
    ["LSTM/GRU", "¿Cuál tiene menos parámetros y suele entrenar más rápido?", ["LSTM", "GRU", "Son idénticas", "Ninguna tiene parámetros"], 1, "Aun así, el paper no concluye cuál es mejor en general."],
    ["LSTM/GRU", "Una red neuronal recursiva opera sobre…", ["secuencias de tiempo únicamente", "entradas estructuradas como árboles / grafos acíclicos", "imágenes RGB", "vectores one-hot"], 1, "Aplica los mismos pesos de forma recursiva sobre la estructura."],
    // ---------- Transfer learning ----------
    ["Transfer learning", "El transfer learning funciona en deep learning si…", ["las características aprendidas en la primera tarea son generales", "las dos tareas no tienen relación", "se entrena desde cero", "no hay datos"], 0, "Deben servir tanto a la tarea base como a la objetivo."],
    ["Transfer learning", "Los tres beneficios del transfer learning son…", ["menos capas, menos datos, menos GPU", "comienzo superior, pendiente más alta y asíntota superior", "mayor learning rate, más dropout, más épocas", "más parámetros, más ruido, más tiempo"], 1, "Se aprecian en la curva de desempeño vs entrenamiento."],
    ["Transfer learning", "El modelo ResNet fue desarrollado por…", ["Oxford", "Google", "Microsoft", "Stanford"], 2, "VGG: Oxford · Inception: Google · ResNet: Microsoft."],
    ["Transfer learning", "GloVe es un modelo de embeddings de…", ["Google", "Stanford", "Facebook", "Microsoft"], 1, "word2vec es de Google."],
    ["Transfer learning", "Al hacer fine-tuning de un modelo preentrenado se recomienda…", ["un learning rate muy alto", "un learning rate muy bajo para no destruir lo aprendido", "reinicializar todos los pesos", "quitar la base convolucional"], 1, "Primero se entrena la cabeza con la base congelada."],
  ];

  const CARDS = [
    ["Neurona", "Unidad básica; un 'estimador débil'. Calcula z = Σwx + b y aplica una activación."],
    ["Función de activación", "Transforma la suma ponderada en la salida; agrega la no linealidad. También 'función de transferencia'."],
    ["Forward propagation", "La entrada viaja en una sola dirección por las capas ocultas hasta la salida."],
    ["Backpropagation", "El error y su gradiente fluyen hacia atrás para actualizar los pesos (regla de la cadena)."],
    ["Learning rate", "Tamaño del paso en cada actualización: muy bajo = lento, muy alto = diverge."],
    ["Batch", "Fragmento aleatorio de igual tamaño de los datos; entrenar por lotes generaliza mejor."],
    ["Época", "Una pasada hacia adelante y hacia atrás de todos los datos."],
    ["Dropout", "Apagar neuronas al azar (p≈0.5) solo en entrenamiento; regularización tipo ensamble."],
    ["Padding", "Orilla de ceros para que la salida tenga el mismo tamaño que la entrada."],
    ["Data augmentation", "Crear datos nuevos derivados de los existentes (rotar, voltear, recortar)."],
    ["Vanishing gradient", "Gradiente muy pequeño → los pesos casi no se actualizan; la red deja de aprender."],
    ["Exploding gradient", "Gradiente muy grande → pesos enormes, inestabilidad, NaN."],
    ["Tensor", "Arreglo n-dimensional (generaliza escalares, vectores y matrices) con sus transformaciones válidas."],
    ["Convolución", "Deslizar un kernel sobre la imagen, multiplicar elemento a elemento y sumar → mapa de características."],
    ["Pooling", "Resumir regiones (max/avg) para reducir dimensiones, parámetros y sobreajuste."],
    ["Flattening", "Convertir la matriz/volumen en un vector para las capas densas."],
    ["Softmax", "Convierte K logits en probabilidades que suman 1; una clase por ejemplo."],
    ["ReLU", "max(0, x). La más usada en ocultas; derivada constante; riesgo de neuronas muertas."],
    ["Perceptrón", "Suma ponderada + umbral → 1 o −1. Aprende con la regla delta."],
    ["Feedforward", "Red sin ciclos: la información solo avanza."],
    ["RNN", "Red con ciclos y 'memoria'; mismos pesos en cada paso de tiempo."],
    ["BPTT", "Backpropagation through time: suma los errores de todos los pasos de tiempo."],
    ["BRNN", "RNN bidireccional: usa información pasada y futura."],
    ["LSTM", "Hochreiter & Schmidhuber. Estado de celda + puertas de olvido, entrada y salida."],
    ["GRU", "Puertas de reinicio y actualización; sin estado separado; menos parámetros."],
    ["Red recursiva", "Mismos pesos aplicados recursivamente sobre un árbol / grafo acíclico."],
    ["Word embedding", "Vector denso y entrenable (8–1024 dims) donde palabras similares quedan cerca."],
    ["Similitud coseno", "a·b / (‖a‖‖b‖): 1 = misma dirección, 0 = ortogonales."],
    ["Transfer learning", "Reutilizar un modelo de una tarea como punto de partida de otra relacionada."],
    ["Transferencia inductiva", "Reducir de forma benéfica el espacio de hipótesis usando un modelo de una tarea relacionada."],
    ["Adam", "Optimizador que combina momentum (1.er momento) y RMSProp (2.º momento)."],
    ["Adadelta", "Variante de Adagrad sin tasa de aprendizaje (Zeiler, 2012)."],
  ];

  const $ = (id) => document.getElementById(id);
  const KEY = "apuntes-quiz-m4";
  let answers = {};
  try { answers = JSON.parse(window.SafeStore ? window.SafeStore.get(KEY) || "{}" : "{}") || {}; } catch (e) { answers = {}; }
  const save = () => { if (window.SafeStore) window.SafeStore.set(KEY, JSON.stringify(answers)); };
  let order = Q.map((_, i) => i), onlyWrong = false;

  // Orden de opciones barajado pero estable por pregunta (así la correcta no cae siempre en la misma letra).
  // data-k guarda el índice original, de modo que las respuestas guardadas no dependen del orden mostrado.
  function perm(i) {
    const p = [0, 1, 2, 3];
    let s = (i + 1) * 2654435761 % 4294967296;
    for (let j = p.length - 1; j > 0; j--) { s = (s * 1664525 + 1013904223) % 4294967296; const r = s % (j + 1); [p[j], p[r]] = [p[r], p[j]]; }
    return p;
  }
  function verdict(i, a, ok) {
    return `<span class="verdict ${a === ok ? "ok" : "bad"}">${a === ok ? "✓ Correcto." : "✗ Incorrecto. La respuesta es " + "ABCD"[perm(i).indexOf(ok)] + "."}</span> `;
  }

  function render() {
    const box = $("quiz"); if (!box) return;
    const tema = $("qz-filter").value;
    const ids = order.filter((i) => (!tema || Q[i][0] === tema) && (!onlyWrong || (answers[i] != null && answers[i] !== Q[i][3])));
    if (!ids.length) { box.innerHTML = `<p class="callout tip">${onlyWrong ? "No hay preguntas falladas con este filtro." : "No hay preguntas con este filtro."}</p>`; updateScore(); return; }
    box.innerHTML = ids.map((i, n) => {
      const [t, q, opts, ok, ex] = Q[i], a = answers[i], done = a != null;
      return `<div class="q${done ? " answered" : ""}" data-i="${i}">
        <div class="q-top"><span class="q-num">Pregunta ${n + 1} de ${ids.length}</span><span class="q-tema">${t}</span></div>
        <div class="q-text">${q}</div>
        <div class="opts">${perm(i).map((k, pos) => `<button class="opt${done && k === ok ? " correct" : ""}${done && k === a && a !== ok ? " wrong" : ""}" type="button" data-k="${k}"${done ? " disabled" : ""}><span class="letter">${"ABCD"[pos]})</span><span>${opts[k]}</span></button>`).join("")}</div>
        <div class="explain">${done ? verdict(i, a, ok) : ""}${ex}</div>
      </div>`;
    }).join("");
    box.querySelectorAll(".opt").forEach((b) => b.addEventListener("click", () => {
      const qi = +b.closest(".q").dataset.i; answers[qi] = +b.dataset.k; save();
      const el = b.closest(".q"), [, , , ok, ex] = Q[qi], a = answers[qi];
      el.classList.add("answered");
      el.querySelectorAll(".opt").forEach((o) => { const k = +o.dataset.k; o.disabled = true; if (k === ok) o.classList.add("correct"); if (k === a && a !== ok) o.classList.add("wrong"); });
      el.querySelector(".explain").innerHTML = verdict(qi, a, ok) + ex;
      updateScore();
    }));
    updateScore();
  }

  function updateScore() {
    const tema = $("qz-filter").value;
    const pool = Q.map((_, i) => i).filter((i) => !tema || Q[i][0] === tema);
    const done = pool.filter((i) => answers[i] != null), good = done.filter((i) => answers[i] === Q[i][3]);
    $("qz-score").textContent = `${good.length} / ${done.length}`;
    $("qz-meter").style.width = (pool.length ? (done.length / pool.length) * 100 : 0) + "%";
    $("qz-score").title = `${done.length} de ${pool.length} respondidas`;
  }

  function cards() {
    const w = $("flash"); if (!w) return;
    w.innerHTML = CARDS.map(([a, b]) => `<div class="flash" tabindex="0" role="button" aria-label="Tarjeta: ${a.replace(/"/g, "&quot;")}"><div class="flash-inner"><div class="flash-face flash-front">${a}</div><div class="flash-face flash-back">${b}</div></div></div>`).join("");
    w.querySelectorAll(".flash").forEach((c) => {
      const flip = () => c.classList.toggle("flipped");
      c.addEventListener("click", flip);
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("quiz")) return;
    const temas = [...new Set(Q.map((q) => q[0]))];
    $("qz-filter").innerHTML = `<option value="">Todos (${Q.length})</option>` + temas.map((t) => `<option value="${t}">${t} (${Q.filter((q) => q[0] === t).length})</option>`).join("");
    document.querySelectorAll("[data-quiz-count]").forEach((el) => { el.textContent = Q.length; });
    $("qz-filter").addEventListener("change", render);
    $("qz-shuffle").addEventListener("click", () => { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } render(); });
    $("qz-wrong").addEventListener("click", () => { onlyWrong = !onlyWrong; $("qz-wrong").classList.toggle("primary", onlyWrong); render(); });
    $("qz-reset").addEventListener("click", () => { answers = {}; save(); order = Q.map((_, i) => i); onlyWrong = false; $("qz-wrong").classList.remove("primary"); render(); });
    render();
    cards();
  });
})();
