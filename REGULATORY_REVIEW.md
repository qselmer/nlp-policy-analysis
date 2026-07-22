# Revisión normativa mundial sobre cambio climático y pesquerías

## Objetivo

Construir un mapa sistemático y auditable de instrumentos jurídicos, políticas y planes de manejo que incorporan cambio climático, variabilidad ambiental, precaución, enfoque ecosistémico o manejo adaptativo en pesquerías marinas, con énfasis en pequeños pelágicos y transferibilidad a la anchoveta peruana.

## Principio metodológico

El sistema no sustituye la interpretación jurídica. Automatiza cuatro tareas:

1. registro y organización de fuentes oficiales;
2. cribado de documentos potencialmente relevantes;
3. extracción estructurada de mecanismos regulatorios con evidencia textual;
4. priorización transparente de casos para revisión humana.

Toda afirmación final debe conservar el artículo o sección exacta, un extracto de respaldo y la URL oficial.

## Estructura inicial

```text
config/
  taxonomy.yml

data/
  raw/                 # documentos oficiales originales, no versionados
  interim/             # texto, fragmentos y predicciones no validadas
  processed/           # tablas finales validadas
  templates/
    source_registry.csv

src/regulatory_review/
  schema.py            # modelos Pydantic y reglas de consistencia
  prompts.py           # prompts restringidos a la evidencia
  scoring.py           # transferibilidad a la anchoveta
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

No escribas una clave API dentro de un notebook. Crea un archivo local `.env`:

```text
OPENAI_API_KEY=your_key_here
```

`.env` está excluido por `.gitignore`.

## Registro de una fuente

1. Descarga el documento desde una fuente oficial.
2. Guarda el original en `data/raw/`.
3. Añade una fila a `data/templates/source_registry.csv`.
4. Conserva la fecha de versión y la fecha de consulta.
5. No sustituyas silenciosamente una versión anterior.

Campos mínimos:

- `document_id` estable y único;
- nombre oficial;
- jurisdicción;
- año y versión;
- tipo y fuerza jurídica;
- autoridad competente;
- URL oficial;
- idioma;
- ruta local.

## Criterios de inclusión

Un documento se incluye cuando cumple simultáneamente:

1. es un instrumento oficial;
2. se aplica a pesca marina, gobernanza pesquera o ecosistemas marinos relevantes;
3. contiene al menos un elemento climático, ambiental, precautorio, ecosistémico o adaptativo.

La literatura académica se conserva en una base separada como evidencia interpretativa, pero no se codifica como norma.

## Unidad de análisis

La unidad primaria no es siempre el documento completo. Para extracción jurídica se utilizará:

- artículo;
- sección;
- anexo;
- medida de conservación;
- regla de control;
- disposición transitoria.

Cada mecanismo regulatorio constituye una fila independiente en la tabla final.

## Validación

Deben revisarse manualmente:

- todos los documentos incluidos;
- todos los instrumentos clasificados como vinculantes;
- todos los extractos y referencias de artículo;
- todos los casos con transferibilidad alta o muy alta;
- una muestra estratificada de documentos excluidos.

El desempeño del cribado debe evaluarse priorizando `recall`, porque omitir una norma relevante es más costoso que revisar un falso positivo.

## Primer corpus prioritario

La búsqueda inicial debe cubrir:

1. instrumentos mundiales de FAO, CMNUCC, CDB y derecho del mar;
2. OROP con medidas precautorias, ecosistémicas o adaptativas;
3. planes de pequeños pelágicos, especialmente anchoveta, sardina, arenque, capelán, caballa, jurel y menhaden;
4. legislación y políticas de Perú, Chile, Estados Unidos, Canadá, Noruega, Australia, Unión Europea, Sudáfrica, Namibia, Nueva Zelanda y Japón.

## Productos finales esperados

- diagrama PRISMA adaptado a normativa;
- tabla mundial de instrumentos;
- matriz de mecanismos y disparadores;
- comparación entre reglas declarativas y operativas;
- ranking razonado de transferibilidad;
- recomendaciones normativas para la anchoveta norte-centro;
- anexo auditable con fuentes, artículos, extractos y decisiones de validación.
