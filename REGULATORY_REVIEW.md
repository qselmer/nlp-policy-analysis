# Revisión integral mundial sobre cambio climático y pesquerías

## Objetivo

Construir un mapa sistemático, comparativo y auditable de la evidencia mundial sobre cambio climático, variabilidad ambiental y adaptación en pesquerías marinas, con énfasis en pequeños pelágicos y en su utilidad para la pesquería norte-centro de anchoveta peruana.

El universo documental comprende, entre otros:

- leyes, reglamentos, acuerdos y medidas de conservación;
- políticas, estrategias, planes de adaptación y planes de manejo;
- protocolos, evaluaciones de stock y asesoramiento científico;
- informes técnicos e institucionales;
- artículos científicos y revisiones;
- capítulos de libro, tesis y documentos metodológicos;
- programas, proyectos piloto y evaluaciones de implementación;
- guías, manuales y marcos conceptuales.

La finalidad no es reunir documentos que solo mencionen el cambio climático. Se busca identificar y comparar evidencia, mecanismos, resultados, proyecciones, lecciones de implementación y propuestas que puedan informar el manejo adaptativo de la anchoveta.

## Principio metodológico

El sistema no sustituye la interpretación científica, pesquera ni jurídica. Automatiza cinco tareas:

1. registro, deduplicación y organización de fuentes;
2. cribado de documentos potencialmente relevantes;
3. clasificación por tipo de fuente y corriente de evidencia;
4. extracción estructurada de hallazgos con localización y respaldo textual;
5. priorización transparente para validación humana y análisis de transferibilidad.

Toda afirmación final debe conservar:

- la fuente primaria o versión consultada;
- la página, artículo, sección, tabla, figura o componente del proyecto;
- un extracto breve o una síntesis verificable;
- la URL, DOI o identificador persistente;
- el estado de validación humana.

## Arquitectura de dos capas

### Capa 1. Mapa general de evidencia

Incluye cualquier fuente que aporte conocimiento relevante sobre:

- impactos y riesgos climáticos para stocks, ecosistemas o pesquerías;
- respuestas biológicas, espaciales, productivas o socioeconómicas;
- métodos de evaluación y proyección;
- opciones de adaptación y manejo;
- desempeño de medidas o proyectos;
- gobernanza, capacidad institucional y barreras de implementación.

### Capa 2. Submódulo normativo y operativo

Se aplica únicamente a leyes, reglamentos, políticas, planes, estrategias, protocolos y medidas de manejo. Evalúa adicionalmente:

- fuerza jurídica;
- autoridad competente;
- principios jurídicos y de manejo;
- indicadores y disparadores;
- tipo de respuesta;
- carácter automático, semiautomático, discrecional o consultivo;
- capacidad de revisión durante la temporada;
- fuerza del asesoramiento científico.

Una fuente científica puede explicar por qué un mecanismo sería útil, pero no debe codificarse como norma. Del mismo modo, una política puede declarar una intención sin aportar evidencia empírica de efectividad.

## Corrientes de evidencia

Cada hallazgo puede pertenecer a una o varias corrientes:

1. **climate_hazard**: cambios o extremos físicos y biogeoquímicos;
2. **ecological_response**: biomasa, productividad, reclutamiento, crecimiento, condición, reproducción, estructura o distribución;
3. **fishery_response**: capturas, esfuerzo, disponibilidad, acceso, seguridad, costos o desempeño;
4. **management_option**: HCR, cuotas, cierres, monitoreo, MSE, alertas y otras respuestas;
5. **governance_and_law**: competencias, mandatos, coordinación, precaución y fuerza normativa;
6. **implementation_project**: acciones, productos, resultados y lecciones de proyectos;
7. **socioeconomic_adaptation**: empleo, cadenas de valor, equidad, seguridad alimentaria y capacidad adaptativa;
8. **methods_and_data**: modelos, indicadores, escenarios, datos y sistemas de observación.

## Estructura del proyecto

```text
config/
  taxonomy.yml

data/
  raw/                    # documentos originales, no versionados
  interim/                # texto, fragmentos y predicciones no validadas
  processed/              # tablas finales validadas
  templates/
    source_registry.csv

src/evidence_review/
  schema.py               # modelos generales de fuente, hallazgo y calidad
  prompts.py              # cribado y extracción integral
  quality.py              # evaluación diferenciada por tipo de fuente
  scoring.py              # transferibilidad a la anchoveta

src/regulatory_review/
  schema.py               # submodelo jurídico-operativo
  prompts.py              # extracción normativa especializada
  scoring.py              # transferibilidad de mecanismos regulatorios
```

## Instalación local

Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-regulatory.txt
```

Linux o macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-regulatory.txt
```

## Credenciales

No escribas claves API dentro de notebooks o scripts versionados. Crea un archivo local `.env`:

```text
OPENAI_API_KEY=your_key_here
```

`.env` está excluido por `.gitignore`.

## Registro de fuentes

1. Identifica la fuente primaria o la mejor versión disponible.
2. Guarda el original en `data/raw/` cuando la licencia y el acceso lo permitan.
3. Añade una fila a `data/templates/source_registry.csv`.
4. Conserva DOI, URL, versión, fecha de consulta y checksum.
5. No sustituyas silenciosamente versiones anteriores.
6. Registra por separado traducciones, anexos o versiones revisadas.

La fuente no tiene que ser oficial en todos los casos. El requisito de oficialidad se aplica a las afirmaciones sobre leyes, políticas, planes gubernamentales y medidas de manejo. Los artículos y capítulos deben vincularse preferentemente a DOI, editor o repositorio institucional.

## Criterios de inclusión

Una fuente se incluye cuando cumple simultáneamente:

1. se refiere a pesca marina, gobernanza pesquera, pequeños pelágicos, peces forrajeros o ecosistemas marinos directamente relevantes;
2. contiene al menos un componente de cambio climático, variabilidad ambiental, eventos extremos, riesgo, precaución, enfoque ecosistémico, adaptación o resiliencia;
3. aporta al menos un hallazgo, mecanismo, método, proyección, evaluación, recomendación o lección de implementación que pueda codificarse.

No se exige que la fuente trate exclusivamente pequeños pelágicos. Se permiten casos indirectos cuando el mecanismo o la evidencia sea razonablemente transferible y esa indirectividad quede registrada.

## Criterios generales de exclusión

- menciones climáticas sin contenido pesquero o marino relevante;
- documentos centrados exclusivamente en acuicultura o aguas continentales, salvo que aporten un mecanismo explícitamente transferible;
- duplicados, resúmenes sin texto suficiente o versiones superadas sin valor histórico;
- opiniones sin método, evidencia o disposición identificable;
- documentos donde no pueda localizarse el hallazgo que se pretende codificar.

## Unidad de análisis

La unidad primaria es un **hallazgo verificable**, no necesariamente el documento completo.

Según la fuente, la localización puede ser:

- artículo, disposición, anexo o medida de conservación;
- sección, párrafo o recomendación;
- método, resultado o discusión;
- tabla, figura o material suplementario;
- objetivo, actividad, producto, resultado o indicador de un proyecto;
- regla de control, protocolo o procedimiento operativo.

Cada hallazgo constituye una fila independiente. Un mismo documento puede generar múltiples filas y corrientes de evidencia.

## Evaluación de calidad diferenciada

No debe aplicarse una única escala a todas las fuentes.

### Artículos, capítulos y tesis

- claridad de la pregunta;
- idoneidad del diseño y de los datos;
- transparencia metodológica;
- tratamiento de incertidumbre;
- validación, sensibilidad y reproducibilidad;
- correspondencia entre resultados y conclusiones.

### Informes técnicos y evaluaciones

- autoridad y trazabilidad de la fuente;
- método y cobertura de datos;
- supuestos y limitaciones;
- revisión técnica o institucional;
- claridad de indicadores y resultados.

### Proyectos y programas

- teoría de cambio y población objetivo;
- línea base;
- indicadores verificables;
- diseño de evaluación;
- evidencia de resultados frente a productos;
- sostenibilidad, escalabilidad y lecciones negativas.

### Normas, políticas y planes

- vigencia, jurisdicción y fuerza jurídica;
- responsabilidades definidas;
- recursos, cronograma y monitoreo;
- indicadores, umbrales y mecanismos de activación;
- conexión entre evidencia y decisión;
- evidencia de implementación y revisión.

La calidad metodológica, la autoridad jurídica y la utilidad para la anchoveta son dimensiones separadas. Una ley vinculante puede carecer de operacionalización; un artículo robusto puede no ser directamente transferible.

## Validación

Deben revisarse manualmente:

- todos los documentos incluidos en la síntesis final;
- todos los extractos, cifras y localizadores;
- todos los instrumentos clasificados como vinculantes;
- todos los resultados cuantitativos usados en conclusiones;
- todos los casos con transferibilidad alta o muy alta;
- una muestra aleatoria estratificada de documentos excluidos;
- una muestra estratificada por tipo de fuente y clase predicha.

El cribado inicial debe priorizar `recall`. La precisión puede recuperarse mediante una segunda fase de revisión, mientras que una fuente relevante omitida no entra en la síntesis.

## Corpus prioritario

La búsqueda inicial cubrirá:

1. instrumentos globales y regionales de pesca, océanos, clima y biodiversidad;
2. OROP y sistemas nacionales con medidas precautorias, ecosistémicas o adaptativas;
3. planes y evaluaciones de pequeños pelágicos, especialmente anchoveta, sardina, arenque, capelán, caballa, jurel y menhaden;
4. artículos, revisiones, informes y capítulos sobre impactos climáticos y adaptación pesquera;
5. proyectos y programas con acciones demostrativas, sistemas de alerta, gobernanza o manejo adaptativo;
6. Perú y otros sistemas de afloramiento, además de casos comparables de Norteamérica, Europa, África, Oceanía y Asia.

## Productos finales esperados

- diagrama PRISMA o ROSES adaptado al universo multifuente;
- tabla maestra de fuentes y decisiones de cribado;
- mapa mundial de evidencia por región, especie y tipo de fuente;
- matriz presión–respuesta–mecanismo–resultado;
- síntesis diferenciada de impactos, opciones, implementación y gobernanza;
- comparación entre propuestas, reglas declarativas y mecanismos operativos;
- evaluación crítica de calidad y vacíos de evidencia;
- ranking razonado de transferibilidad a la anchoveta;
- recomendaciones científicas, operativas e institucionales;
- anexo auditable con fuentes, localizadores, extractos y validación.
