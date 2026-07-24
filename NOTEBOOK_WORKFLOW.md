# Guía de ejecución de notebooks

## Propósito

Esta guía explica qué hace cada notebook, cuándo debe ejecutarse, qué decisiones humanas requiere, qué archivos produce y qué condiciones deben cumplirse antes de avanzar.

El repositorio contiene dos flujos distintos:

1. **Tutorial original de Climate Change AI**
   - `part1_evidence_synthesis.ipynb`
   - `part2_paris_prompts.ipynb`
2. **Extensión para cambio climático y pesquerías marinas**
   - `notebooks/01_...` a `notebooks/13_...`

Los dos notebooks originales son material educativo independiente. No son pasos obligatorios ni entradas del flujo 01–13.

---

## Reglas metodológicas invariantes

Durante todo el flujo deben mantenerse separadas las siguientes categorías:

- propuesta o recomendación ≠ implementación;
- implementación ≠ efectividad;
- actividad ≠ producto ≠ resultado ≠ impacto;
- proyección o escenario ≠ tendencia observada;
- recomendación académica ≠ resultado empírico;
- plan, estrategia o proyecto ≠ norma vinculante;
- número de hallazgos ≠ número de estudios independientes;
- calidad de la fuente ≠ transferibilidad a anchoveta;
- aceptación experta ≠ robustez analítica;
- robustez analítica ≠ eficacia de manejo.

Nunca se deben editar manualmente los productos finales para forzar una conclusión. Las correcciones se hacen en la hoja de revisión correspondiente y luego se regeneran los productos aguas abajo.

---

## Directorios y versionamiento

### Se versiona en Git

- notebooks sin resultados de ejecución;
- código de `src/`;
- archivos de configuración de `config/`;
- pruebas de `tests/`;
- plantillas de `data/templates/`;
- documentación Markdown.

### No se versiona en Git

- documentos descargados;
- exportaciones de bases bibliográficas;
- resultados de `data/raw/`;
- resultados de `data/interim/`;
- productos de `data/processed/`;
- respuestas de modelos y hojas de revisión con datos reales.

Después de ejecutar un notebook:

```powershell
git restore notebooks/<notebook_ejecutado>.ipynb
git status --short
```

El segundo comando debe quedar sin salida antes de actualizar la rama.

---

# Mapa general del flujo

```text
Definir alcance
    ↓
01 Inicializar corpus
    ↓
02 Diseñar y registrar búsquedas
    ↓
03 Importar y deduplicar fuentes
    ↓
04 Cribado de títulos y resúmenes
    ↓
05 Recuperar y registrar textos completos
    ↓
06 Extraer y segmentar documentos
    ↓
07 Cribado a texto completo
    ↓
08 Extraer hallazgos estructurados
    ↓
09 Evaluar calidad de las fuentes
    ↓
10 Evaluar transferibilidad a anchoveta
    ↓
11 Construir apoyo a decisiones
    ↓
12 Auditar robustez y dependencia de fuentes
    ↓
13 Generar tablas, figuras y reportes
```

Los notebooks `01_build_evidence_corpus` y `01_build_regulatory_corpus` son entradas paralelas. El segundo solo es necesario cuando se incorporan instrumentos legales, regulatorios u operativos.

---

# Tutorial original

## `part1_evidence_synthesis.ipynb`

### Función

Tutorial de aprendizaje automático supervisado para identificar y clasificar literatura climática. Presenta modelos basados en conteos, SVM y transformadores sobre un conjunto de datos preparado.

### Cuándo ejecutarlo

- para aprender clasificación supervisada de documentos;
- para comparar enfoques clásicos de NLP con LLM;
- para experimentar con métricas de clasificación;
- para desarrollar posteriormente un modelo de priorización de cribado.

### Relación con el flujo 01–13

Es conceptualmente cercano al notebook 04 porque ambos pueden apoyar la identificación de documentos relevantes. Sin embargo, no sustituye el protocolo de búsqueda, deduplicación, validación humana, extracción de hallazgos, evaluación de calidad ni síntesis.

### No debe usarse como

- revisión sistemática completa;
- evaluación de calidad;
- prueba de implementación o efectividad;
- generador automático de recomendaciones.

---

## `part2_paris_prompts.ipynb`

### Función

Tutorial de clasificación mediante prompts e in-context learning. Trabaja con textos de contribuciones determinadas a nivel nacional del Acuerdo de París y los clasifica según Objetivos de Desarrollo Sostenible.

### Cuándo ejecutarlo

- para aprender diseño de prompts;
- para evaluar clasificación de fragmentos mediante LLM;
- para comparar prompt simple e in-context learning;
- para estudiar un caso de textos de política climática.

### Relación con el flujo 01–13

Es conceptualmente cercano a las fases 07–10 porque usa LLM para clasificar texto. La diferencia es que el flujo de pesquerías exige trazabilidad por fuente y fragmento, taxonomías controladas, validación humana, calidad diferenciada por tipo de fuente, transferibilidad y auditoría de robustez.

### Aclaración sobre “Paris”

`paris` se refiere al **Acuerdo de París** y a documentos NDC. No se refiere a artículos científicos ni a la ciudad de París.

---

# Extensión para clima y pesquerías marinas

## `notebooks/01_build_evidence_corpus.ipynb`

### Objetivo

Inicializar el corpus general de evidencia climática y pesquera y crear las estructuras de datos requeridas por las fases posteriores.

### Ejecutar cuando

- se inicia un proyecto nuevo;
- cambia sustancialmente el alcance;
- se añade una nueva clase de fuente no contemplada;
- se reconstruyen plantillas o taxonomías.

### Trabajo humano

- confirmar pregunta y alcance;
- revisar tipos de fuente incluidos;
- revisar regiones, ecosistemas, especies y pesquerías;
- comprobar vocabularios de `config/taxonomy.yml`.

### Salidas principales

- registros y plantillas iniciales del corpus;
- estructura de carpetas;
- metadatos mínimos de trazabilidad.

### Criterio para avanzar

- alcance documentado;
- identificadores únicos disponibles;
- taxonomías compatibles con las fuentes previstas;
- ninguna fuente real sobrescrita por una plantilla vacía.

---

## `notebooks/01_build_regulatory_corpus.ipynb`

### Objetivo

Inicializar el subcorpus de leyes, reglamentos, protocolos, planes de manejo y otros instrumentos con autoridad jurídica u operativa.

### Ejecutar cuando

- se incorporan normas o instrumentos de manejo;
- se comparan jurisdicciones;
- se estudian disparadores, respuestas, autoridad o fuerza jurídica.

### Trabajo humano

- identificar jurisdicción y autoridad emisora;
- distinguir norma vinculante, política, plan, protocolo y propuesta;
- registrar vigencia, alcance y nivel institucional;
- no inferir obligatoriedad desde el título del documento.

### Criterio para avanzar

- cada instrumento tiene jurisdicción y tipo documental;
- la fuerza jurídica no se confunde con calidad metodológica;
- las versiones derogadas o reemplazadas están señaladas.

---

## `notebooks/02_search_strategy.ipynb`

### Objetivo

Diseñar y registrar una búsqueda reproducible, multibase y multilingüe.

### Ejecutar cuando

- se crea la búsqueda inicial;
- se agrega una base de datos;
- se amplían regiones, idiomas, especies o tipos de fuente;
- se realiza una actualización temporal.

### Trabajo humano

- revisar bloques conceptuales;
- adaptar sintaxis por base;
- documentar fecha, plataforma, consulta y filtros;
- registrar búsquedas fallidas y cambios de estrategia;
- evitar filtrar por anchoveta demasiado pronto cuando el objetivo es aprender de otras pesquerías.

### Salidas principales

- registro de búsquedas;
- consultas reproducibles;
- plan de cobertura por base, idioma y tipo de fuente.

### Criterio para avanzar

- cada búsqueda tiene fecha, base y cadena exacta;
- la cobertura incluye literatura científica y gris pertinente;
- la estrategia permite identificar evidencia negativa, nula y contextual, no solo ejemplos exitosos.

---

## `notebooks/03_import_and_deduplicate_sources.ipynb`

### Objetivo

Importar exportaciones heterogéneas, normalizar metadatos y deduplicar fuentes.

### Ejecutar cuando

- llegan nuevos archivos de Scopus, Web of Science, OpenAlex u otras bases;
- se agregan referencias manuales;
- se incorporan informes, proyectos o instrumentos institucionales.

### Trabajo humano

- revisar mapeos de columnas;
- comprobar DOI, URL, título, año y autores;
- auditar pares marcados como posibles duplicados;
- conservar versiones distintas cuando representan documentos sustantivamente diferentes.

### Salidas principales

- registro único de fuentes;
- manifiesto de importación;
- tabla de duplicados y decisiones.

### Criterio para avanzar

- no existen identificadores duplicados no explicados;
- cada fuente conserva procedencia de importación;
- la deduplicación no eliminó informes o versiones institucionales legítimas.

---

## `notebooks/04_screen_titles_and_abstracts.ipynb`

### Objetivo

Realizar cribado de alta sensibilidad sobre título y resumen.

### Ejecutar cuando

- termina la importación y deduplicación;
- se añaden nuevas fuentes al registro;
- se recalibra el criterio de inclusión.

### Trabajo humano

- revisar inclusiones, exclusiones y casos inciertos;
- documentar una razón controlada de exclusión;
- priorizar sensibilidad sobre especificidad en esta fase;
- no excluir únicamente porque una fuente no menciona anchoveta.

### Salidas principales

- decisiones de título/resumen;
- subconjunto candidato a texto completo;
- métricas de avance y desacuerdo.

### Criterio para avanzar

- todos los registros tienen decisión o estado pendiente explícito;
- las exclusiones tienen razón;
- los casos inciertos se mantienen para texto completo.

---

## `notebooks/05_retrieve_and_register_full_texts.ipynb`

### Objetivo

Localizar, descargar y registrar textos completos sin perder la relación con la fuente bibliográfica.

### Ejecutar cuando

- finaliza el cribado inicial;
- se reciben PDFs adicionales;
- cambia el acceso o se reemplaza una versión incompleta.

### Trabajo humano

- comprobar que el archivo corresponde a la fuente;
- registrar acceso, licencia y versión;
- identificar suplementos;
- documentar textos no recuperados y motivos.

### Salidas principales

- inventario de documentos;
- estado de recuperación;
- ruta o referencia local del texto completo.

### Criterio para avanzar

- cada fuente incluida tiene texto, estado no recuperado o justificación;
- no hay PDFs asociados a la fuente incorrecta;
- los documentos institucionales conservan URL y fecha de acceso.

---

## `notebooks/06_parse_and_segment_documents.ipynb`

### Objetivo

Extraer texto y dividir documentos en unidades auditables para cribado y extracción.

### Ejecutar cuando

- se añaden o reemplazan textos completos;
- cambia el parser;
- la extracción de texto anterior fue incompleta.

### Trabajo humano

- revisar PDFs escaneados o con texto defectuoso;
- comprobar encabezados, páginas, tablas y referencias;
- marcar documentos que necesitan tratamiento manual;
- evitar interpretar texto desordenado como evidencia.

### Salidas principales

- texto extraído;
- segmentos con localizador;
- diagnóstico de calidad de parseo.

### Criterio para avanzar

- cada segmento conserva `source_id` y localizador;
- los errores de extracción están identificados;
- el cuerpo del documento puede distinguirse de referencias y anexos.

---

## `notebooks/07_screen_full_texts.ipynb`

### Objetivo

Aplicar los criterios definitivos de elegibilidad usando el documento completo.

### Ejecutar cuando

- finaliza el parseo;
- se corrige un documento;
- se amplía el alcance de elegibilidad.

### Trabajo humano

- verificar población, exposición, respuesta y contexto;
- excluir con razones específicas;
- diferenciar mención incidental de evidencia utilizable;
- mantener fuentes contextuales cuando cumplen una función definida.

### Salidas principales

- fuentes incluidas a texto completo;
- fuentes excluidas con razón;
- registro de incertidumbre y revisión.

### Criterio para avanzar

- todas las fuentes recuperadas tienen decisión;
- las exclusiones son reproducibles;
- las fuentes incluidas contienen al menos una unidad potencialmente extraíble o un vacío explícito.

---

## `notebooks/08_extract_structured_findings.ipynb`

### Objetivo

Extraer hallazgos a nivel de afirmación o resultado, con respaldo textual y trazabilidad.

### Ejecutar cuando

- el conjunto de textos incluidos está cerrado para una iteración;
- se incorporan nuevas fuentes elegibles;
- se corrige la segmentación o taxonomía.

### Trabajo humano

- validar cada hallazgo asistido;
- corregir tipo de hallazgo y corriente de evidencia;
- comprobar fragmento y localizador;
- separar resultados empíricos, proyecciones, recomendaciones, reglas y brechas;
- rechazar hallazgos que exceden lo respaldado por la fuente.

### Salidas principales

- hallazgos aceptados, corregidos, rechazados y pendientes;
- diagnósticos de consistencia;
- vínculo entre hallazgo, segmento y fuente.

### Criterio para avanzar

- cero problemas estructurales;
- todos los hallazgos utilizados están aceptados o corregidos;
- ningún resumen afirma más que su fragmento de respaldo.

---

## `notebooks/09_appraise_evidence_quality.ipynb`

### Objetivo

Evaluar la calidad o autoridad de cada fuente mediante instrumentos apropiados para su tipo documental.

### Ejecutar cuando

- se valida el conjunto de fuentes con hallazgos;
- se incorpora una fuente nueva;
- se corrige su clasificación documental.

### Trabajo humano

- revisar dominio de evaluación aplicable;
- corregir respuestas y limitaciones críticas;
- no aplicar una sola escala idéntica a artículos, políticas, leyes y proyectos;
- mantener separada calidad metodológica de fuerza legal.

### Salidas principales

- evaluaciones de calidad validadas;
- puntuaciones normalizadas y clases;
- vínculo hallazgo–calidad.

### Criterio para avanzar

- todas las fuentes con hallazgos tienen evaluación validada;
- ningún hallazgo queda sin calidad asociada;
- las limitaciones críticas están documentadas.

---

## `notebooks/10_synthesise_transferability.ipynb`

### Objetivo

Evaluar cómo puede utilizarse cada hallazgo para la anchoveta norte-centro.

### Ejecutar cuando

- hallazgos y calidad están validados;
- se agregan nuevas fuentes;
- cambia el perfil de la pesquería objetivo.

### Trabajo humano

- revisar similitud ecológica;
- revisar relevancia climática y de manejo;
- revisar especificidad operacional y factibilidad de datos en Perú;
- corregir uso propuesto y papel en la síntesis;
- no convertir una propuesta externa en una recomendación directa para Perú.

### Salidas principales

- evaluación de transferibilidad por hallazgo;
- matriz de síntesis;
- resumen temático;
- opciones de manejo, brechas y candidatos preliminares.

### Criterio para avanzar

- todas las evaluaciones utilizadas están completadas y validadas;
- cero problemas de validación;
- la fortaleza de evidencia coincide con la calidad validada;
- transferibilidad alta no se interpreta como efectividad.

---

## `notebooks/11_build_decision_support.ipynb`

### Objetivo

Agrupar evidencia transferible en cadenas climáticas, opciones de manejo y recomendaciones condicionadas.

### Ejecutar cuando

- la fase 10 está cerrada;
- cambian las evaluaciones de transferibilidad;
- se amplía el conjunto de medidas.

### Trabajo humano

- revisar las recomendaciones candidatas;
- aceptar, corregir o rechazar cada una;
- evaluar requisitos de datos, instituciones y adaptación local;
- conservar lenguaje condicional.

### Salidas principales

- cadenas presión–respuesta–pesquería–manejo;
- portafolio de opciones;
- madurez operacional;
- prioridades de investigación;
- recomendaciones validadas y pendientes.

### Criterio para avanzar

- todas las recomendaciones tienen revisión experta;
- cero recomendaciones pendientes;
- cero problemas estructurales;
- una opción implementada no se declara efectiva sin evaluación.

---

## `notebooks/12_audit_robustness.ipynb`

### Objetivo

Evaluar estabilidad frente a filtros alternativos, cambios de umbrales y eliminación secuencial de fuentes.

### Ejecutar cuando

- las recomendaciones de la fase 11 están validadas;
- se agregan fuentes nuevas;
- cambian umbrales de madurez o reglas de retención.

### Trabajo humano

- revisar escenarios;
- interpretar dependencia de fuentes;
- distinguir hallazgos de fuentes independientes;
- no ocultar recomendaciones sensibles;
- comprobar que estabilidad no se presenta como eficacia.

### Salidas principales

- resultados por escenario;
- análisis leave-one-source-out;
- concentración de fuentes;
- independencia de evidencia;
- estabilidad de recomendaciones.

### Criterio para avanzar

- todos los escenarios configurados están representados;
- cada recomendación tiene clase de estabilidad;
- cero problemas de validación;
- las recomendaciones sensibles permanecen visibles.

---

## `notebooks/13_build_scientific_products.ipynb`

### Objetivo

Generar tablas, figuras y reportes reproducibles a partir de resultados ya validados.

### Ejecutar cuando

- la fase 12 tiene cero problemas;
- el conjunto de datos está congelado para una entrega;
- se necesita regenerar material científico después de una actualización.

### Trabajo humano

- revisar legibilidad de figuras;
- comprobar coherencia entre tablas y narrativa;
- revisar que las categorías de publicación reflejen estabilidad;
- evitar lenguaje causal en redes de coocurrencia;
- comprobar que ninguna conclusión supera la evidencia.

### Salidas principales

- tablas de evidencia, portafolio, recomendaciones y brechas;
- seis figuras;
- reporte de síntesis;
- métodos y resultados;
- resumen ejecutivo;
- manifiesto de productos y problemas.

### Criterio de cierre

- todos los productos configurados existen y tienen contenido;
- cero problemas de publicación;
- figuras revisadas visualmente;
- tablas y reportes reproducen los mismos conteos;
- los notebooks se restauran antes de hacer `git pull` o commit.

---

# Cómo ampliar el corpus con nuevas fuentes

Sí, el sistema está diseñado para ampliarse. La fase desde la cual se reinicia depende de cómo entra la nueva evidencia.

## Ruta A: nueva base bibliográfica o nueva búsqueda

```text
02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

Ejemplos:

- Scopus;
- Web of Science;
- OpenAlex;
- Dimensions;
- Lens;
- Google Scholar como búsqueda complementaria documentada;
- ASFA;
- repositorios institucionales;
- literatura en español, portugués o francés.

## Ruta B: referencias encontradas por búsqueda de citas

```text
03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

Registrar que la procedencia fue búsqueda hacia atrás, hacia adelante o recomendación experta.

## Ruta C: informe o PDF conocido y ya disponible

```text
03 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

No debe copiarse directamente a la carpeta de documentos sin crear antes su registro de fuente.

## Ruta D: nueva ley, reglamento o instrumento de manejo

```text
01_build_regulatory_corpus → 03 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

Cuando existe título y resumen utilizable también puede pasar por el notebook 04.

## Ruta E: corregir una fuente ya incorporada

- metadatos incorrectos: reiniciar en 03;
- PDF incorrecto: reiniciar en 05;
- texto mal extraído: reiniciar en 06;
- decisión de elegibilidad: reiniciar en 07;
- hallazgo mal codificado: reiniciar en 08;
- calidad incorrecta: reiniciar en 09;
- transferibilidad incorrecta: reiniciar en 10.

Después de cualquier corrección deben regenerarse todas las fases posteriores.

---

# Estrategia recomendada para ampliar sin perder la versión actual

## 1. Congelar la línea base

Antes de agregar fuentes, conservar una copia local de:

```text
data/raw/
data/interim/
data/processed/
```

Ejemplo:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
Compress-Archive `
  -Path data\raw, data\interim, data\processed `
  -DestinationPath "..\nlp_policy_analysis_baseline_$stamp.zip" `
  -Force
```

El archivo debe guardarse fuera del repositorio porque los datos y productos están excluidos de Git.

## 2. Definir un identificador de actualización

Ejemplo:

```text
update_2026_01
update_2026_02
```

Registrar:

- fecha de cierre de búsqueda;
- bases consultadas;
- idiomas;
- periodo cubierto;
- número de registros importados;
- responsable;
- versión de configuración y commit.

## 3. Mantener dos líneas de trabajo

### Línea congelada

Usada para:

- manuscrito;
- informe técnico;
- material suplementario;
- resultados citables.

### Línea viva

Usada para:

- nuevas fuentes;
- búsquedas actualizadas;
- cambios de taxonomía;
- pruebas metodológicas.

No debe cambiarse una tabla del manuscrito con resultados de la línea viva sin declarar una nueva versión analítica.

---

# Fuentes que conviene ampliar

La ampliación debe priorizar independencia y vacíos, no solo aumentar el número de hallazgos.

## Prioridad alta

- evaluaciones de implementación y efectividad;
- estudios independientes sobre medidas actualmente respaldadas por una sola fuente;
- pesquerías de pequeños pelágicos en sistemas de afloramiento;
- documentos de manejo adaptativo con reglas operacionales explícitas;
- MSE y harvest control rules sensibles al clima;
- sistemas de alerta temprana y evaluación de riesgo climático;
- evidencia institucional de Perú, Chile, California, Benguela y Canarias;
- literatura gris de FAO, ICES, NOAA, CSIRO, OECD, Banco Mundial y organismos pesqueros regionales;
- normas, planes y protocolos oficiales con versión y vigencia verificables.

## Evitar como estrategia principal

- aumentar múltiples hallazgos provenientes del mismo documento;
- buscar únicamente medidas exitosas;
- incorporar opiniones sin respaldo identificable;
- tratar resúmenes de proyectos como evaluaciones de impacto;
- agregar una norma sin verificar jurisdicción, vigencia y autoridad.

---

# Puntos de validación humana

| Fase | Decisión humana principal |
|---|---|
| 03 | confirmar duplicados dudosos |
| 04 | incluir, excluir o mantener incierto |
| 05 | confirmar correspondencia fuente–archivo |
| 06 | confirmar calidad del texto extraído |
| 07 | decisión definitiva a texto completo |
| 08 | aceptar, corregir o rechazar hallazgos |
| 09 | aceptar o corregir calidad de fuente |
| 10 | aceptar o corregir transferibilidad |
| 11 | aceptar, corregir o rechazar recomendaciones |
| 12 | revisar interpretación de sensibilidad |
| 13 | revisar coherencia científica y visual |

La automatización asiste, pero no reemplaza estas decisiones.

---

# Comandos de control

## Antes de ejecutar

```powershell
git status --short
git pull --ff-only origin agent/regulatory-review-scaffold
$env:PYTHONPATH = 'src'
pytest -q
```

## Después de ejecutar

```powershell
git restore notebooks/<notebook_ejecutado>.ipynb
git status --short
```

## Regla de avance

No avanzar al siguiente notebook cuando exista alguno de estos estados:

- problemas de validación mayores que cero;
- registros pendientes que alimentan el siguiente paso;
- identificadores duplicados;
- fuentes sin trazabilidad;
- hallazgos sin respaldo textual;
- recomendaciones sin revisión experta;
- discrepancias entre tablas, figuras y reportes.

---

# Respuesta directa: ¿es parecido al tutorial original?

**Sí, pero solo en el nivel general.** Ambos usan NLP para organizar grandes colecciones de texto climático.

El tutorial original enseña dos técnicas concretas:

1. clasificación supervisada de literatura climática;
2. clasificación con prompts de textos NDC del Acuerdo de París según ODS.

La extensión 01–13 convierte esa idea educativa en un sistema de revisión de evidencia completo y auditable para pesquerías marinas. Añade búsqueda reproducible, deduplicación, cribado en dos etapas, recuperación y parseo de texto completo, extracción por hallazgo, calidad por tipo de fuente, transferibilidad a anchoveta, revisión experta, apoyo a decisiones, sensibilidad, dependencia de fuentes y productos científicos.

Por tanto:

- `part1` y `part2` sirven para aprender y experimentar;
- `notebooks/01–13` sirven para ejecutar el estudio;
- los resultados del tutorial original no se mezclan automáticamente con el corpus de pesquerías;
- las técnicas del tutorial sí pueden reutilizarse posteriormente para mejorar priorización, clasificación o extracción, siempre con validación humana.
