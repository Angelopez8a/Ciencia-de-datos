/* =========================================================
   quiz-m1.js · cuestionario y tarjetas de repaso del Módulo 1
   Cada pregunta: [tema, pregunta, [opciones], índice correcto, explicación]
   Las respuestas numéricas y de código se verificaron ejecutándolas (Python 3.13, pandas 3.0, sklearn 1.9)
   ========================================================= */
(function () {
  "use strict";
  const Q = [
    // ---------- Conceptos ----------
    ["Conceptos", "Un tablero con el promedio mensual de afluencia de cada estación responde a la pregunta \"¿qué pasó?\". ¿Qué término lo describe mejor?", ["Machine Learning", "Business Intelligence", "Big Data", "Deep Learning"], 1, "BI es descriptivo: reportes y tableros sobre lo que ya ocurrió. Los cubos OLAP de la sesión 10 son BI."],
    ["Conceptos", "Según la clase, la Inteligencia Artificial es…", ["un conjunto de redes neuronales profundas", "sistemas computacionales que simulan comportamientos cognitivos humanos", "cualquier modelo estadístico", "el análisis de Big Data"], 1, "Es la definición de la diapositiva 3 de la sesión 2. ML está dentro de IA y DL dentro de ML."],
    ["Conceptos", "Las \"3 V\" de Big Data son…", ["validez, valor y veracidad", "volumen, velocidad y variedad", "varianza, valor y volumen", "visualización, velocidad y validación"], 1, "Volumen, velocidad y variedad: cuando rebasan las herramientas tradicionales se necesita cómputo distribuido (Hadoop, Spark…)."],
    ["Conceptos", "¿Cuál es el orden correcto de los primeros cinco pasos del proceso de la sesión 2?", ["Recolección, limpieza, integración, transformación, exploración", "Recolección, integración, limpieza, exploración, transformación", "Integración, recolección, exploración, limpieza, transformación", "Exploración, recolección, integración, limpieza, modelación"], 1, "1 Recolección, 2 Integración, 3 Limpieza, 4 Exploración, 5 Transformación; después modelación, validación, producción y calibración."],
    ["Conceptos", "El paso 9 del proceso, \"Calibración\", significa…", ["elegir el learning rate", "monitorear el modelo en producción y re-entrenarlo cuando se degrada", "estandarizar las variables", "limpiar los datos otra vez antes de modelar"], 1, "El proceso es un ciclo: tras producción el modelo se monitorea y se recalibra (en el Módulo 2: reportes de estabilidad, PSI)."],
    ["Conceptos", "¿Cuál de estos es un <strong>hiperparámetro</strong> (decisión del analista) y no un parámetro aprendido?", ["coef_ de la regresión logística", "intercept_ de la regresión lineal", "el tamaño de la ventana de observación (vobs = 20)", "la mediana que guarda SimpleImputer"], 2, "coef_, intercept_ y la mediana se aprenden con fit. vobs, el umbral KS o el número de bines los fija el analista."],
    ["Conceptos", "En SEXO = {Hombre, Mujer}, según la diapositiva de credit scoring…", ["Hombre y Mujer son características", "SEXO es la característica y Hombre/Mujer son sus atributos", "SEXO es la variable objetivo", "Hombre es la unidad muestral"], 1, "Característica (feature) = la variable; atributos = sus valores. En la scorecard cada atributo recibe puntos."],
    ["Conceptos", "¿Cuál de estas columnas <strong>nunca</strong> debe entrar como predictora?", ["Monthly_Charge", "Contract", "Customer_ID", "Tenure_in_Months"], 2, "Es la llave: identifica la unidad muestral y sirve para unir tablas, pero no describe al cliente."],

    // ---------- TAD y ventanas ----------
    ["TAD y ventanas", "La unidad muestral se define como…", ["el resultado de una observación", "la representación numérica de un objeto", "la columna con más varianza", "el conjunto de validación"], 1, "Variable objetivo = resultado de una observación. Unidad muestral = representación numérica de un objeto (un renglón de la TAD)."],
    ["TAD y ventanas", "En Ecobici los datos vienen a nivel viaje, pero la pregunta es por estación. ¿Qué implica?", ["Que no se puede modelar", "Que hay que cambiar la granularidad: agregar a nivel estación-periodo", "Que hay que usar deep learning", "Que la unidad muestral es el viaje"], 1, "La unidad muestral la define el problema, no el archivo: estación + periodo (ancla)."],
    ["TAD y ventanas", "Con t de 1 a 700, vobs = 20 y vdes = 1, ¿cuántas anclas hay por estación?", ["700", "699", "680", "20"], 2, "anclai = 1 + 20 − 1 = 20; anclaf = 700 − 1 = 699; de 20 a 699 hay 680 anclas (× 390 estaciones = 265,200 renglones)."],
    ["TAD y ventanas", "Con 36 periodos, vobs = 12 y vdes = 3, ¿cuál es la primera y la última ancla válida?", ["12 y 33", "1 y 36", "12 y 36", "13 y 33"], 0, "anclai = t_min + vobs − 1 = 12; anclaf = t_max − vdes = 33 (22 anclas)."],
    ["TAD y ventanas", "¿Qué imprime?<pre>df = pd.DataFrame({\"t\": [1, 2, 3, 4, 5]})\nvobs, ancla = 2, 4\nprint(df[(df.t &gt; ancla - vobs) &amp; (df.t &lt;= ancla)].t.tolist())</pre>", ["[2, 3, 4]", "[3, 4]", "[4, 5]", "[3, 4, 5]"], 1, "La ventana de observación es (ancla − vobs, ancla] = (2, 4] → t = 3 y 4. Es el filtro del notebook de la sesión 8."],
    ["TAD y ventanas", "Para construir y con ancla = 50 y vdes = 1, se toma…", ["la afluencia en t = 50", "la afluencia en t = 51", "el promedio de t = 31 a 50", "la afluencia en t = 49"], 1, "y sale de la ventana de desempeño: el futuro inmediato después del ancla."],
    ["TAD y ventanas", "Una variable de X calculada con datos posteriores al ancla provoca…", ["multicolinealidad", "data leakage: excelente en entrenamiento, falla en producción", "subajuste", "una distribución alterada en el KS"], 1, "\"Ojo con variables futuras\": en producción esa información todavía no existiría."],
    ["TAD y ventanas", "¿Qué imprime?<pre>l = [1, 3, 2, 4, 5]\nprint(\"\".join(str(int(y &gt; x)) for x, y in zip(l, l[1:])))</pre>", ["\"1011\"", "\"0100\"", "\"1101\"", "\"11011\""], 0, "1→3 sube (1), 3→2 baja (0), 2→4 sube (1), 4→5 sube (1). Con split('0') salen rachas de 1 y 2: max_racha_inc = 2."],
    ["TAD y ventanas", "Para la serie [5, 7, 7, 6, 9, 10], sum_inc y sum_dec valen…", ["3 y 2", "3 y 1", "4 y 1", "2 y 1"], 1, "Cambios: +2, 0, −1, +3, +1 → 3 subidas, 1 bajada y 1 empate. Por eso sum_inc + sum_dec no siempre es n − 1."],

    // ---------- pandas ----------
    ["pandas", "¿Qué devuelve <code>pd.Series([1, 2, 2, 3, None]).describe()[\"count\"]</code>?", ["5.0", "4.0", "3.0", "1.0"], 1, "count no cuenta los nulos. Por eso 1 − count / len da el porcentaje de ausentes."],
    ["pandas", "El notebook calcula nulos con <code>1 - df.describe().T[\"count\"] / len(df)</code>. ¿Qué limitación tiene?", ["No funciona con floats", "describe() solo incluye columnas numéricas: las categóricas quedan fuera", "Cuenta los ceros como nulos", "Ninguna"], 1, "Para todas las columnas se usa df.isna().mean(). En churn las categóricas se trataron aparte (fillna \"Sin categoría\")."],
    ["pandas", "<code>df.groupby(\"Contract\")[\"Churn_Value\"].mean()</code> devuelve…", ["el número de clientes por contrato", "la tasa de fuga (proporción de 1) por contrato", "la suma de cargos", "el WoE de cada contrato"], 1, "La media de una variable 0/1 es una proporción."],
    ["pandas", "La tabla izquierda tiene la llave A dos veces y la derecha una vez. Un <code>merge</code> inner por esa llave…", ["falla", "devuelve una fila por cada combinación: A aparece 2 veces", "elimina los duplicados", "devuelve una sola fila de A"], 1, "merge hace todas las combinaciones. validate=\"one_to_one\" lanza MergeError y evita duplicar filas sin darte cuenta."],
    ["pandas", "Diferencia entre <code>how=\"inner\"</code> y <code>how=\"left\"</code>:", ["ninguna", "inner conserva solo llaves presentes en ambas; left conserva todas las de la izquierda (con NaN donde no hay match)", "left elimina nulos", "inner ordena por la llave"], 1, "En el notebook de churn se usa left para pegar las categóricas a las filas que sobrevivieron a la limpieza."],
    ["pandas", "<code>pivot_table(index=\"id_estacion\", columns=\"t\", values=\"afluencia\")</code> convierte la tabla…", ["de ancho a largo", "de largo a ancho: una fila por estación y una columna por periodo", "a una tabla de frecuencias", "a un cubo de 3 dimensiones"], 1, "Formato largo = una fila por estación y periodo; ancho = una columna por periodo."],
    ["pandas", "¿Qué hace <code>reduce(f, [df1, df2, df3])</code>?", ["f(df1) + f(df2) + f(df3)", "f(f(df1, df2), df3)", "concatena los tres", "elimina duplicados"], 1, "Aplica f de dos en dos acumulando: así se integran varias tablas con un solo merge repetido."],
    ["pandas", "En la tabla de frecuencias, FRA es…", ["frecuencia absoluta", "frecuencia relativa acumulada (la última siempre vale 1)", "frecuencia relativa", "frecuencia absoluta acumulada"], 1, "FA absoluta, FR relativa, FAA absoluta acumulada, FRA relativa acumulada."],

    // ---------- Integración ----------
    ["Integración", "En churn se descartan Churn_Label, Churn_Score, Churn_Reason y Churn_Category porque…", ["tienen muchos nulos", "se derivan de la fuga: darían la respuesta (data leakage)", "son categóricas", "tienen baja varianza"], 1, "Solo se conserva Churn_Value como variable objetivo."],
    ["Integración", "Tenure_in_Months y Tenure_Months obtuvieron exactamente el mismo RS_Own en VarClus. ¿Qué indica?", ["Un error de VarClus", "Que son la misma información con otro nombre (duplicado de la integración)", "Que ambas deben entrar al modelo", "Que tienen varianza 0"], 1, "Vienen de archivos distintos. VarClus los pone juntos y solo uno se queda."],
    ["Integración", "Un merge ingenuo de dos tablas que comparten la columna Count además de la llave produce…", ["un error", "las columnas Count_x y Count_y", "una sola columna Count", "filas duplicadas"], 1, "Por eso se usa una función que quita las columnas repetidas antes de unir."],

    // ---------- Limpieza ----------
    ["Limpieza", "En churn solo el 3 % de las filas estaba completo. ¿Qué se hizo con los nulos?", ["Eliminar las filas con nulos", "Imputar con la media (SimpleImputer)", "Eliminar todas las variables", "Reemplazar por cero"], 1, "Regla de la clase: si son pocos, eliminar registros; si no, imputar. Eliminar habría dejado 213 de 7,043 clientes."],
    ["Limpieza", "¿Para qué se calcula KS entre la variable original (sin nulos) y la imputada?", ["Para medir el poder predictivo", "Para detectar si la imputación alteró la distribución; KS &gt; 0.1 → variable \"rota\"", "Para elegir el número de bines", "Para estandarizar"], 1, "Muchos nulos imputados con un solo valor crean un pico artificial."],
    ["Limpieza", "En churn, Total_Revenue (50 % nulos) tuvo KS = 0.303 y Monthly_Charges (10 %) KS = 0.056. ¿Qué se hizo?", ["Se quedaron ambas", "Total_Revenue se descartó; Monthly_Charges se conservó", "Se descartaron ambas", "Se imputaron otra vez"], 1, "El umbral es 0.1. A más nulos imputados, más se deforma la distribución."],
    ["Limpieza", "<code>SimpleImputer(strategy=\"median\")</code> sobre [1, NaN, 3, 10] devuelve…", ["[1, 4.67, 3, 10]", "[1, 3, 3, 10]", "[1, 0, 3, 10]", "[1, 1, 3, 10]"], 1, "La mediana de {1, 3, 10} es 3. Con la media sería 4.67: la mediana es menos sensible al 10."],
    ["Limpieza", "¿Por qué en Ecobici el umbral de VarianceThreshold fue 0.1 y en churn 1.0?", ["Por error", "Porque la varianza depende de la escala de cada variable y se aplica antes de estandarizar", "Porque churn tiene menos filas", "Porque 1.0 es el valor por defecto"], 1, "Una proporción entre 0 y 1 nunca pasa de 0.25 de varianza; un cargo en dólares tiene varianzas enormes."],
    ["Limpieza", "Antes de eliminar los registros con extremos, el notebook insiste en…", ["estandarizar", "pegar la unidad muestral (y el ancla) a X", "calcular el IV", "imputar otra vez"], 1, "Si no, después no se puede unir X con y."],
    ["Limpieza", "Un registro se marca como extremo si…", ["su valor es negativo", "alguna de sus variables está fuera de los percentiles 1 y 99", "tiene nulos", "su KS &gt; 0.1"], 1, "Se crea un indicador ol_ por variable y ext = máximo de ellos: basta un extremo para marcar la fila."],

    // ---------- VarClus ----------
    ["VarClus", "RS_Ratio se calcula como…", ["RS_Own / RS_NC", "(1 − RS_Own) / (1 − RS_NC)", "RS_Own − RS_NC", "1 − RS_Own · RS_NC"], 1, "Entre más bajo, mejor representa la variable a su propio cluster y menos se parece al siguiente."],
    ["VarClus", "Variable A: RS_Own = 0.90, RS_NC = 0.20. Variable B: RS_Own = 0.95, RS_NC = 0.50. ¿Cuál se elige?", ["A (RS_Ratio 0.125)", "B (RS_Ratio 0.100)", "Las dos", "Ninguna"], 1, "A: 0.10 / 0.80 = 0.125. B: 0.05 / 0.50 = 0.100. Gana el RS_Ratio más bajo."],
    ["VarClus", "Si el negocio no permite quedarse con la variable id == 1 de un cluster…", ["se borra el cluster completo", "se toma al menos una variable por cluster", "se usan todas", "se repite VarClus"], 1, "\"Lo ideal es quedarse con el id == 1, pero si el negocio no lo permite, al menos una por cluster\"."],
    ["VarClus", "VarClus se usa en el curso para atacar…", ["el desbalance de clases", "la multicolinealidad (variables redundantes)", "los valores nulos", "las categorías raras"], 1, "En Ecobici v_sum y v_mean tienen correlación 1: llevan la misma información."],

    // ---------- Estandarización ----------
    ["Estandarización", "¿Qué valores da StandardScaler a [1, 2, 3]?", ["[−1, 0, 1]", "[−1.2247, 0, 1.2247]", "[0, 0.5, 1]", "[−0.5, 0, 0.5]"], 1, "Usa la desviación poblacional (ddof = 0): σ = 0.8165, y 1/0.8165 = 1.2247."],
    ["Estandarización", "MinMaxScaler sobre [10, 20, 30, 40, 1000]: ¿cuánto vale el 20?", ["0.25", "0.0101", "0.5", "0.02"], 1, "(20 − 10) / (1000 − 10) = 0.0101: un solo extremo comprime a todos los demás cerca de 0."],
    ["Estandarización", "La ventaja de Min-Max según la diapositiva es que…", ["da media 0", "mantiene la proporción entre valores", "elimina extremos", "no necesita fit"], 1, "Si un valor es el doble que otro (respecto al mínimo), lo sigue siendo."],
    ["Estandarización", "¿Con qué datos se debe hacer el <code>fit</code> del StandardScaler?", ["Con todos los datos", "Solo con entrenamiento; luego transform en entrenamiento y validación", "Solo con validación", "Da igual"], 1, "Mismo principio que el WoE en churn: lo que se \"aprende\" sale solo del 70 %."],

    // ---------- PCA y t-SNE ----------
    ["PCA y t-SNE", "¿Por qué hay que estandarizar antes de PCA?", ["Porque PCA solo acepta valores entre 0 y 1", "Porque PCA se basa en la covarianza y las variables con valores grandes dominarían la varianza total", "Para eliminar nulos", "No hace falta"], 1, "Es la primera viñeta de la diapositiva de reducción de dimensiones."],
    ["PCA y t-SNE", "Con el dataset Wine estandarizado, ¿cuántos componentes se necesitan para ≥ 80 % de varianza?", ["2", "3", "5", "13"], 2, "Acumulada: 36.2, 55.4, 66.5, 73.6, 80.2 → 5 de 13."],
    ["PCA y t-SNE", "¿Por qué t-SNE no se usa para alimentar un modelo predictivo?", ["Porque es lineal", "No conserva la estructura global y no puede transformar datos nuevos (no tiene .transform)", "Porque es muy rápido", "Porque requiere variables categóricas"], 1, "Es para visualizar clústeres en 2D/3D. Además es costoso y depende del perplexity."],

    // ---------- Categóricas ----------
    ["Categóricas", "<code>pd.get_dummies(pd.Series([\"a\",\"b\",\"c\",\"a\"]), drop_first=True).shape</code> es…", ["(4, 3)", "(4, 2)", "(3, 2)", "(4, 1)"], 1, "3 categorías → 2 columnas con drop_first (la categoría eliminada queda como referencia)."],
    ["Categóricas", "Tres variables con 3, 4 y 10 categorías. ¿Cuántas columnas dummy con drop_first=True?", ["17", "14", "3", "10"], 1, "(3 − 1) + (4 − 1) + (10 − 1) = 14. Sin drop_first serían 17: así \"explota\" la dimensión."],
    ["Categóricas", "Frecuencias: A 48 %, B 30 %, C 16 %, D 4 %, E 2 %. Con normalización al 3 %, ¿qué queda?", ["A, B, C, D, E", "A, B, C, D y Otros (con E)", "A, B, C y Otros", "A y Otros"], 1, "Solo E (2 %) está por debajo de 3 %. Ojo: \"Otros\" queda con 2 %; si sigue siendo pequeño se puede juntar con la siguiente."],
    ["Categóricas", "En churn, Zip_Code tenía cientos de valores y después de normalizar se eliminó. ¿Por qué?", ["Por tener nulos", "Todos sus valores tenían &lt; 3 %, se volvieron \"Otros\" y la variable quedó unaria", "Por KS", "Por IV infinito"], 1, "Unaria = una sola categoría: no aporta información."],
    ["Categóricas", "¿Por qué la diapositiva dice que las dummies causan multicolinealidad si no se manejan?", ["Porque son binarias", "Porque las k columnas suman 1 en cada fila: una es combinación lineal de las demás (trampa de la dummy)", "Porque tienen nulos", "Porque se correlacionan con y"], 1, "Se resuelve eliminando una columna (drop_first=True)."],

    // ---------- WoE e IV ----------
    ["WoE e IV", "WoE<sub>i</sub> se define como…", ["ln(%evento / %no evento)", "ln(%no evento / %evento)", "%no evento − %evento", "eventos / total"], 1, "Con esta definición, WoE &gt; 0 = bin con más \"buenos\" que el promedio."],
    ["WoE e IV", "Bin A: 30 eventos y 170 no eventos; bin B: 70 eventos y 730 no eventos. ¿WoE de A?", ["0.1473", "−0.4626", "0.4626", "−0.1473"], 1, "%evento A = 30/100 = 0.30; %no evento A = 170/900 = 0.1889; ln(0.1889 / 0.30) = −0.4626."],
    ["WoE e IV", "Con los mismos bines (A: 30/170, B: 70/730), el IV vale ≈ 0.068. ¿Qué poder predictivo tiene?", ["No ayuda", "Bajo", "Regular", "Fuerte"], 1, "0.02 ≤ IV &lt; 0.1 → bajo."],
    ["WoE e IV", "n_Customer_Status tuvo IV = ∞. ¿Qué lo provoca?", ["Demasiados bines", "Un atributo (Churned) con 0 no eventos: ln(0 / p) → −∞", "Muchos nulos", "Que es numérica"], 1, "Todos los Churned son fuga: la variable da la respuesta y no puede entrar al modelo."],
    ["WoE e IV", "En la sesión 9, ¿qué filtro de IV se aplicó para elegir variables?", ["IV &gt; 0.02", "0.1 &lt; IV &lt; 0.8", "IV &gt; 0.5", "IV &lt; 0.3"], 1, "Por eso n_Contract (1.506) quedó fuera aunque es muy predictiva. \"No son regla dura\"."],
    ["WoE e IV", "¿Por qué el IV tiende a subir al aumentar el número de bines?", ["Por un error de cálculo", "Más cortes separan mejor eventos y no eventos en los datos de entrenamiento (riesgo de sobreajuste)", "Porque disminuye la muestra", "No sube nunca"], 1, "Por eso el notebook limita a 2–6 cortes por cuantiles."],
    ["WoE e IV", "El mapa WoE se calcula con…", ["todos los datos", "solo entrenamiento, y se aplica a entrenamiento y validación", "solo validación", "una muestra aleatoria del 10 %"], 1, "\"Se toma el 70 % para entrenar (o generar WoE) y el 30 % para validar\"."],
    ["WoE e IV", "¿Cuál es una ventaja de WoE frente a dummies?", ["Es más interpretable", "Reduce la dimensionalidad: una sola columna por variable, y captura la relación con el objetivo", "Sirve para cualquier modelo", "No requiere discretizar"], 1, "Desventajas: menos interpretable, requiere discretizar/segmentar y no es útil para todos los modelos."],

    // ---------- Scorecard ----------
    ["Scorecard", "Con PDO = 20, ¿cuánto vale el Factor?", ["20", "13.86", "28.85", "40"], 2, "Factor = PDO / ln 2 = 20 / 0.6931 = 28.85."],
    ["Scorecard", "PDO = 20 y score 600 para momios 50:1. ¿Qué score corresponde a momios 100:1?", ["610", "620", "640", "1200"], 1, "Duplicar los momios suma PDO puntos: 600 + 20 = 620. (Y momios 25:1 → 580.)"],
    ["Scorecard", "En la scorecard, ¿qué hace el Offset?", ["Cambia la dispersión de los scores", "Desplaza toda la distribución de scores sin cambiar su forma", "Elimina extremos", "Duplica los momios"], 1, "El Factor estira o encoge; el Offset mueve."],
    ["Scorecard", "\"Pablito, 28 años, gana $26,500, vive en Monterrey\": 15 + 42 + … + 33. ¿Qué es cada sumando?", ["Un WoE", "Los puntos del atributo correspondiente en cada característica", "Un coeficiente β", "Una probabilidad"], 1, "La scorecard asigna puntos por atributo; el score total es la suma (350 en el ejemplo)."],

    // ---------- Logística ----------
    ["Logística", "Los momios (odds) de un evento con probabilidad p son…", ["p", "1 − p", "p / (1 − p)", "ln(p)"], 2, "La logística es lineal en ln(p / (1 − p))."],
    ["Logística", "Si logit = −2 + 0.8·x y x = 1.5, ¿cuál es P(evento)?", ["0.80", "0.45", "0.31", "0.69"], 2, "logit = −0.8; p = 1 / (1 + e^{0.8}) = 0.310. Los momios son e^{−0.8} = 0.449."],
    ["Logística", "En Ecobici, β de ss_v_max_afluencia = 3.244. Subir 1 desviación estándar…", ["suma 3.244 a la probabilidad", "multiplica los momios por e^{3.244} ≈ 25.6", "multiplica la probabilidad por 3.244", "no cambia nada"], 1, "Suma β al logit, que equivale a multiplicar los momios por e^β."],
    ["Logística", "En la logística de Ecobici el intercepto es −3.229. Para una estación con todas las z = 0, P(alta) ≈", ["0.50", "0.038", "0.96", "0.23"], 1, "1 / (1 + e^{3.229}) = 0.038: la clase positiva es minoritaria."],
    ["Logística", "Con variables WoE, los coeficientes de la logística suelen salir negativos porque…", ["hay un error", "WoE alto = más no eventos, así que a más WoE menor probabilidad de evento", "la sigmoide es decreciente", "se estandarizó"], 1, "En el notebook de churn 16 de 20 coeficientes fueron negativos."],

    // ---------- Métricas ----------
    ["Métricas", "Con TN = 80, FP = 20, FN = 10, TP = 40, el recall es…", ["0.667", "0.80", "0.727", "0.90"], 1, "Recall = TP / (TP + FN) = 40 / 50 = 0.80. Precisión = 40/60 = 0.667; F1 = 0.727."],
    ["Métricas", "En churn la clase positiva es 27 %. Un modelo que dice \"nadie se va\" tendría accuracy de…", ["27 %", "50 %", "73 %", "100 %"], 2, "Acierta todos los negativos. Por eso el accuracy de 0.806 no impresiona y se miran AUC, recall y F1."],
    ["Métricas", "¿Qué usa roc_auc_score?", ["la clase predicha con corte 0.5", "las probabilidades (predict_proba[:, 1])", "los coeficientes", "la matriz de confusión"], 1, "accuracy y confusion_matrix usan predict (clase); AUC usa probabilidades."],
    ["Métricas", "En sklearn, confusion_matrix(y_real, y_pred) para binario se ordena como…", ["[[TP, FP], [FN, TN]]", "[[TN, FP], [FN, TP]]", "[[TP, FN], [FP, TN]]", "[[FN, TP], [TN, FP]]"], 1, "Filas = real (0, 1), columnas = predicción (0, 1)."],
    ["Métricas", "La métrica que se reportó para la regresión de Ecobici fue…", ["AUC", "MAE ≈ 27,006", "Accuracy", "IV"], 1, "Error absoluto medio: en promedio el pronóstico se equivoca por unas 27 mil afluencias."],

    // ---------- OLAP ----------
    ["OLAP", "Registrar cada medición horaria de cada estación es un sistema…", ["OLAP", "OLTP", "de BI", "de cubos"], 1, "OLTP: transacciones del día a día, detalle crudo, muchas inserciones."],
    ["OLAP", "Pasar de promedio por estación a promedio por zona es…", ["drill-down", "roll-up", "slice", "pivot"], 1, "Roll-up sube de nivel de agregación; drill-down baja."],
    ["OLAP", "Quedarse solo con el trimestre Q1 de un cubo es…", ["dice", "slice", "roll-up", "pivot"], 1, "Slice fija una dimensión; dice forma un subcubo con varias condiciones."],
    ["OLAP", "En la sesión 10, ¿cómo se define la \"zona\" de una estación?", ["Por alcaldía", "Por el cuadrante respecto a la longitud y latitud promedio de las estaciones", "Por altitud", "Por K-means"], 1, "Centro: longitud −99.1359, latitud 19.4092 → Zona 1 a Zona 4."],
    ["OLAP", "¿Por qué se hace hrs = hrs − 1 en el notebook de contaminantes?", ["Por el horario de verano", "Porque el archivo registra horas de 1 a 24 y se necesitan de 0 a 23 para crear la fecha", "Para centrar la variable", "Por error"], 1, "Así 24:00 se vuelve 23 y pd.to_datetime funciona."],
  ];

  const CARDS = [
    ["Unidad muestral", "Representación numérica de un objeto: lo que es cada renglón de la TAD (estación + ancla, Customer_ID)."],
    ["Variable objetivo", "Resultado de una observación; lo que se predice (y). Churn_Value, afluencia en t+1."],
    ["TAD", "Tabla Analítica de Datos: matriz de predictoras X + vector solución y, un renglón por unidad muestral."],
    ["Ancla", "Momento t que separa la ventana de observación (X) de la de desempeño (y)."],
    ["Anclas válidas", "De t_min + vobs − 1 a t_max − vdes. Ecobici: 20 a 699 → 680 por estación."],
    ["Característica vs atributo", "SEXO es la característica; Hombre y Mujer son sus atributos."],
    ["Data leakage", "Usar en X información no disponible al predecir: variables futuras o que dan la respuesta (Customer_Status)."],
    ["Granularidad", "Nivel de detalle de una tabla. Viaje → estación-periodo requiere agregar."],
    ["reduce + merge", "Integra varias tablas aplicando el merge de dos en dos por la llave."],
    ["validate=\"one_to_one\"", "Hace que merge falle si la llave está duplicada en alguna tabla."],
    ["% de nulos", "1 − count / len (solo numéricas con describe) o df.isna().mean() (todas)."],
    ["¿Eliminar o imputar?", "Pocos nulos → eliminar registros; muchos → imputar (media/mediana con SimpleImputer)."],
    ["KS > 0.1", "La imputación alteró la distribución: variable \"rota\", se descarta."],
    ["VarianceThreshold", "Quita variables casi constantes. El umbral depende de la escala (0.1 Ecobici, 1.0 churn)."],
    ["Extremos", "Registro fuera de p1–p99 en alguna variable. Antes de borrar, pegar la unidad muestral."],
    ["RS_Own", "R² de la variable con su propio cluster. Alto = bien representada."],
    ["RS_NC", "R² con el cluster más cercano que no es el suyo. Bajo = no se confunde."],
    ["RS_Ratio", "(1 − RS_Own)/(1 − RS_NC). Más bajo = mejor representante; se toma id == 1."],
    ["Z-score", "(x − media)/desviación. Media 0, desviación 1 (sklearn usa ddof = 0)."],
    ["Min-Max", "(x − min)/(max − min). Entre 0 y 1; conserva proporciones; sensible a extremos."],
    ["fit vs transform", "fit aprende (mediana, media, WoE) solo con train; transform lo aplica a train y validación."],
    ["PCA", "Combinaciones lineales de máxima varianza; basada en covarianza → estandarizar antes."],
    ["t-SNE", "No lineal, para visualizar en 2D/3D. No conserva estructura global; no sirve para predecir."],
    ["Normalización (categóricas)", "Agrupar categorías con frecuencia < 3 % en \"Otros\"."],
    ["Unaria", "Variable con una sola categoría: no aporta y se elimina."],
    ["Trampa de la dummy", "Las k dummies suman 1 → multicolinealidad. Solución: drop_first=True."],
    ["WoE", "ln(%no evento / %evento). Positivo = menos riesgo que el promedio."],
    ["IV", "Σ(%no evento − %evento)·WoE. <0.02 nada · 0.02–0.1 bajo · 0.1–0.3 regular · 0.3–0.5 fuerte · ≥0.5 sobrepredictiva."],
    ["IV = ∞", "Un atributo sin eventos o sin no eventos: casi siempre la variable da la respuesta."],
    ["Momios", "p / (1 − p). La logística es lineal en ln(momios)."],
    ["e^β", "Factor por el que se multiplican los momios al subir una unidad en x."],
    ["PDO", "Points to Double the Odds: puntos que duplican los momios bueno:malo."],
    ["Factor y Offset", "Factor = PDO/ln 2 (escala). Offset = Score − Factor·ln(odds) (desplaza)."],
    ["Puntos por atributo", "(−WoE·β + α/n)·Factor + Offset/n (fórmula de la diapositiva)."],
    ["Recall vs precisión", "Recall = TP/(TP+FN): de los positivos reales, cuántos detecto. Precisión = TP/(TP+FP)."],
    ["Accuracy engañoso", "Con 27 % de positivos, decir \"todos negativos\" da 73 % de accuracy."],
    ["OLTP vs OLAP", "OLTP registra transacciones (detalle); OLAP analiza agregados históricos en cubos."],
    ["Roll-up / drill-down", "Subir de nivel (estación → zona) / bajar (zona → estación → hora)."],
    ["Slice / dice", "Fijar una dimensión (solo Q1) / subcubo con varias condiciones."],
  ];

  const $ = (id) => document.getElementById(id);
  const KEY = "apuntes-quiz-m1";
  let answers = {};
  try { answers = JSON.parse(window.SafeStore ? window.SafeStore.get(KEY) || "{}" : "{}") || {}; } catch (e) { answers = {}; }
  const save = () => { if (window.SafeStore) window.SafeStore.set(KEY, JSON.stringify(answers)); };
  let order = Q.map((_, i) => i), onlyWrong = false;

  // Orden de opciones barajado pero estable por pregunta (para que la correcta no caiga siempre en la misma letra).
  // data-k guarda el índice original, así las respuestas guardadas no dependen del orden mostrado.
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
    if (!ids.length) { box.innerHTML = `<p class="callout tip">${onlyWrong ? "¡No tienes preguntas falladas con este filtro! 🎉" : "No hay preguntas con este filtro."}</p>`; updateScore(); return; }
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
    document.querySelectorAll("[data-card-count]").forEach((el) => { el.textContent = CARDS.length; });
    $("qz-filter").addEventListener("change", render);
    $("qz-shuffle").addEventListener("click", () => { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } render(); });
    $("qz-wrong").addEventListener("click", () => { onlyWrong = !onlyWrong; $("qz-wrong").classList.toggle("primary", onlyWrong); render(); });
    $("qz-reset").addEventListener("click", () => { answers = {}; save(); order = Q.map((_, i) => i); onlyWrong = false; $("qz-wrong").classList.remove("primary"); render(); });
    render();
    cards();
  });
})();
