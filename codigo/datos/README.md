# Datos reales usados en los ejemplos

Todos los archivos son copias **sin modificar** de fuentes públicas (en el caso de GloVe, un subconjunto de renglones copiados tal cual). Cada programa intenta leer primero la copia
local y, si no existe (por ejemplo, cuando se descarga un solo `.py`), la descarga desde la dirección original
que aparece en su código.

| Archivo | Contenido | Fuente | Se usa en |
|---|---|---|---|
| `telco_churn_ibm.csv` | 7,043 clientes de una compañía telefónica: servicios contratados, cargos, antigüedad y si abandonaron la compañía (`Churn`). Es el caso de la sesión 9. | IBM, repositorio [telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (licencia Apache 2.0). | `m1_00b`, `m1_12`, `m1_13`, `m1_14`, `m1_15`, `m2_02`, `m2_07`, `m2_10` |
| `credito_upc.csv` | 4,454 solicitudes de crédito con su desenlace (`good` / `bad`), antigüedad laboral, vivienda, edad, estado civil, ingresos, gastos, activos, deuda y monto solicitado. Contiene valores faltantes reales. | Curso de minería de datos de la Universitat Politècnica de Catalunya (T. Aluja, L. A. Belanche), vía [gastonstat/CreditScoring](https://github.com/gastonstat/CreditScoring) y el paquete de R `modeldata` (`credit_data`). | `m1_06`, `m1_07`, `m1_08` |
| `calidad_aire_nueva_york_1973.csv` | 153 mediciones diarias de ozono, radiación solar, viento y temperatura en Nueva York (mayo a septiembre de 1973). | Departamento de Conservación del Estado de Nueva York y Servicio Meteorológico Nacional de EE. UU.; publicado en Chambers et al. (1983), *Graphical Methods for Data Analysis*. Conjunto `airquality` de R. | `m1_03` |
| `gapminder.csv` | Esperanza de vida, población y PIB per cápita de 142 países, cada cinco años de 1952 a 2007. | [Gapminder](https://www.gapminder.org/data/) (CC BY 4.0), vía el paquete de R `gapminder`. | `m1_00a`, `m1_01`, `m1_04`, `m2_02`, `m2_10` (PSI) |
| `galton_familias.csv` | Estaturas (en pulgadas) de 934 hijos y de sus padres en 205 familias, registradas por Francis Galton. | Galton (1886); paquete de R `HistData` (`GaltonFamilies`). | `m2_01`, `m2_02`, `m2_05`, `m2_10` |
| `challenger_juntas.csv` | Temperatura y número de juntas tóricas (*O-rings*) dañadas en los 23 lanzamientos del transbordador espacial previos al accidente del Challenger (28 de enero de 1986). | Comisión Presidencial sobre el Accidente del Transbordador Challenger (1986); Dalal, Fowlkes y Hoadley (1989). Paquete de R `openintro` (`orings`). | `m2_03` |
| `old_faithful.csv` | Duración de 272 erupciones del géiser Old Faithful (Yellowstone) y tiempo de espera hasta la siguiente, en minutos. | Härdle (1991); Azzalini y Bowman (1990). Conjunto `faithful` de R. | `m3_02` |
| `sismos_fiji.csv` | Ubicación (latitud, longitud), profundidad y magnitud de 1,000 sismos de magnitud mayor a 4.0 cerca de Fiyi, desde 1964. | Proyecto PRIM-H de la Universidad de Harvard (J. Woodhouse). Conjunto `quakes` de R. | `m3_04` |
| `mayorista_lisboa.csv` | Gasto anual de 440 clientes de un distribuidor mayorista de Portugal en seis categorías de producto, con su canal (hotel/restaurante/café o minorista). | UCI Machine Learning Repository, *Wholesale customers* (Cardoso, 2014; CC BY 4.0), vía el repositorio [udacity/machine-learning](https://github.com/udacity/machine-learning). | `m3_05` |
| `taxis_nyc_2019_03.csv` | 6,433 viajes de taxi de la ciudad de Nueva York en marzo de 2019: horarios, distancia, tarifa, propina, forma de pago y zonas de origen y destino. | Registros de viajes de la NYC Taxi & Limousine Commission, vía el repositorio [seaborn-data](https://github.com/mwaskom/seaborn-data). | `m1_10` |
| `glove_6B_50d_subconjunto.csv` | Vectores de 50 dimensiones de 39 palabras en inglés, copiados sin cambios del modelo GloVe 6B (400,000 palabras, entrenado con Wikipedia 2014 y Gigaword 5). | Pennington, Socher y Manning (2014), Stanford NLP (licencia PDDL), vía [gensim-data](https://github.com/RaRe-Technologies/gensim-data). | `m4_09` y el simulador de embeddings del Módulo 4 |

Otros conjuntos reales se cargan directamente desde las librerías: `load_wine`, `load_iris`, `load_breast_cancer`,
`load_diabetes` y `load_digits` (scikit-learn); la concentración de CO₂ en Mauna Loa (`statsmodels`); y, en Keras,
las 50,000 reseñas de películas de IMDB (Maas et al., 2011) y los 70,000 dígitos escritos a mano de MNIST
(LeCun, Cortes y Burges). Las coordenadas de las seis ciudades de `m3_03` son las del centro de cada ciudad,
redondeadas a centésimas de grado.
