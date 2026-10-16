# ⚡ Analista de Datos (`analista-de-datos`)

> **Motor Neuro-Simbólico de Razonamiento Cognitivo, Auditoría Forense y Descomposición Financiera de Alto Volumen.**  
> *Diseñado para operar como biblioteca Python, CLI de terminal o microservicio API REST de nivel empresarial.*

[![GitHub Repo](https://img.shields.io/badge/GitHub-anthra123x%2Fanalista--de--datos-blue?logo=github)](https://github.com/anthra123x/analista-de-datos)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-brightgreen)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20Vector-FFF000?logo=duckdb)](https://duckdb.org/)
[![Polars](https://img.shields.io/badge/Polars-High%20Performance-CD792C?logo=polars)](https://pola.rs/)

---

## 🚀 ¿Por qué Analista de Datos?

Los LLMs autorregresivos convencionales (*GPT-4o, Claude 3.5, Gemini*) alucinan sumas decimales en un **13% - 25%** de los casos, saturan su ventana de contexto con tablas grandes y son incapaces de auditar distribuciones estadísticas masivas o sistemas matemáticos insolubles.

**Analista de Datos** resuelve esto mediante una arquitectura **Neuro-Simbólica**:
1. **Front-End Cognitivo (LLM):** Extrae la semántica, entidades, notas al pie y conceptos contables no estructurados (formato ChatML con traza explícita `<think>...</think>`).
2. **Núcleo Simbólico Determinista (DuckDB + Polars):** Ejecuta álgebra exacta a velocidad nativa C++ (>900 documentos/segundo, 0.00% error aritmético).
3. **Solucionador de Sistemas Incompatibles (Rouché-Frobenius):** Detecta contradicciones matemáticas en facturas alteradas y proyecta la perturbación mínima $L_2$ mediante pseudoinversas de Moore-Penrose.
4. **Auditoría Forense Integrada:**
   - **Ley de Benford:** Detección de manipulación contable mediante el índice MAD de Nigrini.
   - **Detección de Smurfing / Structuring:** Alerta ante fraccionamiento deliberado de pagos.
   - **Checksum Fiscal de NIF/CIF/NIE:** Validación algorítmica de los dígitos de control de la AEAT.
   - **Puntualidad Temporal:** Normalización ISO-8601, cálculo de plazos de pago y días de demora en mora.
   - **Asientos Contables PGC:** Generación automática de asientos de partida doble (Debe/Haber).

---

## 📊 Benchmark Comparativo (2.500 Documentos Complejos)

| Métrica | Pure LLM (GPT-4o) | Frontier CoT (DeepSeek-R1) | **Analista de Datos (Ours)** |
| :--- | :--- | :--- | :--- |
| **Throughput** | 10.5 docs/s | 0.55 docs/s | **922.5 docs/s** |
| **Tiempo Total (2.500 docs)** | 237.5 s | 4.531.2 s (~1.25 h) | **2.71 s** |
| **Latencia p50** | 820 ms | 12.400 ms | **1.08 ms** |
| **Error Aritmético / Alucinación** | 13.44% | 3.50% | **0.00% (CERO - Determinista)** |
| **Sistemas Incompatibles** | Alucina descuentos | Solo explicación textual | **Pseudoinversa Moore-Penrose ($L_2$)** |
| **Ley de Benford (Forense)** | No soportado | No soportado | **Soportado (MAD Nigrini: 0.0072)** |
| **Conciliación Bancaria** | Inviable (>contexto) | Inviable por tokens | **100% Integrado (Sub-segundo)** |
| **Coste por cada 10.000 Facturas** | ~$125.00 USD | ~$1.200.00 USD | **$0.00 USD (Inferencia Local)** |
| **Consumo RAM Pico** | ~45.0 MB | ~110.0 MB | **33.8 MB** |

---

## 📦 Instalación

### Opción 1: Vía Git / Pip
```bash
git clone https://github.com/anthra123x/analista-de-datos.git
cd analista-de-datos
pip install -e .
```

### Opción 2: Con Gestor Ultrarrápido `uv`
```bash
uv pip install -e .
```

---

## 🛠️ Guía Rápida de Uso

### 1. Como Biblioteca en Python (SDK)

```python
from analista_de_datos import AnalistaDeDatos

analista = AnalistaDeDatos()

factura_texto = """
PROVEEDOR: Telefónica de España S.A.U.
NIF: A28015865
FACTURA: TEL-2026-8819
FECHA: 15 de marzo de 2026
VENCIMIENTO: 60 días
Fibra Óptica 1Gbps + Centralita Cloud | 1 x 120.00€ = 120.00€ (IVA 21%)
Subtotal: 120.00€
IVA 21%: 25.20€
TOTAL: 145.20€
"""

# Analizar documento
doc = analista.analizar(factura_texto)

print(f"Factura: {doc.invoice_number}")
print(f"Total: {doc.grand_total} {doc.currency}")
print(f"Partida Doble: {'Balanceada' if doc.is_balanced else 'Descuadrada'}")
print(f"Score de Precisión: {doc.composite_precision_score}")
print(f"NIF Válido: {doc.tax_id_validation['is_valid']} ({doc.tax_id_validation['entity_type']})")
print(f"Fecha Normalizada: {doc.temporal_analysis['normalized_invoice_date']}")

# Asiento Contable PGC
if doc.journal_entry:
    print("\n--- Asiento Contable Generado ---")
    for linea in doc.journal_entry.lines:
        print(f"  [{linea.account_code}] {linea.account_name} | Debe: {linea.debit}€ | Haber: {linea.credit}€")
```

---

### 2. Conciliación Bancaria Automática

```python
from analista_de_datos import AnalistaDeDatos

analista = AnalistaDeDatos()

movimientos_banco = [
    {"date": "2026-03-20", "concept": "TRF PAGO PROV TELEFONICA", "amount": -145.20},
    {"date": "2026-03-22", "concept": "COMISION BANCARIA MANTENIMIENTO", "amount": -15.00}
]

facturas = [
    {"invoice_number": "TEL-2026-8819", "grand_total": 145.20, "supplier_name": "Telefónica"}
]

reconciliacion = analista.conciliar_banco(movimientos_banco, facturas, tolerance_cents=0.05)
print(f"Reconciliado: {reconciliacion.reconciled_percentage}% ({reconciliacion.matched_count}/{reconciliacion.total_bank_movements})")
print(f"Movimientos Huérfanos: {len(reconciliacion.unreconciled_movements)}")
```

---

### 3. Como Herramienta de Línea de Comandos (CLI)

```bash
# 1. Analizar un documento o factura individual
analista analyze factura.txt

# 2. Analizar y obtener JSON completo
analista analyze factura.txt --json

# 3. Procesar un directorio con 10.000 facturas en lote
analista batch ./carpeta_facturas/ --output reporte_consolidado.json

# 4. Validar matemáticamente un NIF/NIE/CIF
analista validate-tax-id A28015865

# 5. Generar dataset sintético para Fine-Tuning (ChatML <think>)
analista generate-dataset --count 100 --output ./dataset.jsonl

# 6. Iniciar el servidor API REST
analista serve --host 0.0.0.0 --port 8100
```

---

### 4. Vía Docker / Docker Compose

```bash
# Levantar el microservicio API
docker compose up -d

# Consultar salud del servicio
curl http://localhost:8100/health
```

---

### 5. API REST Endpoints (OpenAPI / Swagger)

Al ejecutar `analista serve`, la documentación interactiva Swagger UI estará disponible en `http://localhost:8100/docs`.

#### `POST /api/v1/decompose`
Analiza un documento de texto y devuelve la traza CoT junto al JSON estructurado.

```bash
curl -X POST http://localhost:8100/api/v1/decompose \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "PROVEEDOR: AWS EMEA NIF: LU26375245 TOTAL: 100.00 USD",
    "document_name": "aws_factura.txt",
    "use_llm": false
  }'
```

#### `POST /api/v1/finance/reconcile-bank`
Conciliación bidireccional entre extractos bancarios y facturas del libro mayor.

#### `POST /api/v1/finance/prorata-amortization`
Cálculo de la Prorrata General de IVA (Art. 104 LIVA) y cuadro de amortización de activos.

#### `POST /api/v1/finance/fraud-audit`
Auditoría forense de fraccionamiento deliberado (*Smurfing*), importes redondos y concentración de proveedores.

#### `POST /api/v1/jobs/submit`
Encola hasta 50.000+ documentos para procesamiento asíncrono desacoplado con telemetría en streaming Server-Sent Events (SSE) vía `/api/v1/jobs/{job_id}/stream`.

---

## 🧪 Pruebas y Validación

El proyecto incluye suites completas de pruebas unitarias, de integración, de estrés y de benchmarks:

```bash
# Ejecutar suite de pruebas de precisión y puntualidad
python -m tests.test_precision_and_punctuality

# Ejecutar pruebas de endpoints financieros
python -m tests.test_financial_endpoints

# Ejecutar benchmark comparativo frente a modelos frontier
python -m tests.benchmark_comparative

# Ejecutar prueba de álgebra incompatible (Rouché-Frobenius)
python -m tests.test_unsolvable_math

# Ejecutar prueba de estrés con 5.000 facturas y colas SSE
python -m tests.advanced_stress_suite
```

---

## 🏛️ Licencia

Distribuido bajo la Licencia **MIT**. Consulta [LICENSE](LICENSE) para más información.
Desarrollado con arquitectura neuro-simbólica para entornos corporativos y analítica de datos a gran escala.
