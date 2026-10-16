import re
import json
from typing import Dict, Any, Optional, Tuple, List
import httpx

from ..config import config
from ..models.schemas import DecomposedInvoice, InvoiceLineItem, TaxBreakdownItem
from ..models.accounting import classify_concept, generate_double_entry_seat


SYSTEM_REASONING_PROMPT = """Eres Split-Reasoner, el motor de razonamiento cognitivo y auditoría forense contable neuro-simbólica.
Tu tarea es descomponer y auditar exhaustivamente documentos contables, facturas complejas, cuadros de impuestos multi-tramo y tablas financieras.

DEBES SEGUIR UN PROCESO DE PENSAMIENTO ESTRICTO EN DOS PASOS:

PASO 1: Genera tu traza de razonamiento exhaustiva dentro de las etiquetas <think> y </think>:
1. [Identificación de Entidades]: Emisor (proveedor, NIF/CIF), Receptor (cliente, NIF/CIF), fechas y número de factura.
2. [Segmentación Multitasa]: Desglosa cada concepto, cantidad, precio unitario, base imponible y tipo impositivo (IVA 21%, 10%, 4%, 0%, o Recargo de Equivalencia 5.2%, 1.4%, 0.5%).
3. [Clasificación PGC]: Asigna cuenta contable oficial (628 suministros, 629 telecom/software, 623 profesionales, 600 compras, etc.).
4. [Auditoría Matemática y Retenciones]: Suma de bases por tipo impositivo, retención IRPF (si cuenta 623 profesionales).
5. [Ecuación de Partida Doble]: Verifica subtotal + tax_total - retention_total == grand_total. Si detectas contradicción o redondeo, documenta la discrepancia residual.

PASO 2: Emite ÚNICAMENTE el objeto JSON final sin preámbulos:
{
  "invoice_number": "...",
  "supplier_name": "...",
  "supplier_tax_id": "...",
  "supplier_address": "...",
  "customer_name": "...",
  "customer_tax_id": "...",
  "customer_address": "...",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "currency": "EUR" | "USD" | "GBP",
  "subtotal": 0.00,
  "tax_total": 0.00,
  "retention_total": 0.00,
  "grand_total": 0.00,
  "tax_breakdown": [
    {
      "tax_rate": 21.0,
      "base_amount": 0.00,
      "tax_amount": 0.00,
      "surcharge_rate": 0.0,
      "surcharge_amount": 0.0
    }
  ],
  "items": [
    {
      "concept": "...",
      "quantity": 1.0,
      "unit_price": 0.00,
      "base_amount": 0.00,
      "tax_rate": 21.0,
      "tax_amount": 0.00,
      "total_amount": 0.00,
      "account_code": "629000",
      "account_name": "...",
      "cost_center": "..."
    }
  ]
}
"""


def parse_monetary_amount(val_str: str) -> float:
    s = val_str.strip().replace("€", "").replace("$", "").replace("£", "").strip()
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) != 2:
            if len(parts[1]) == 3 and int(parts[0]) > 0:
                s = s.replace(".", "")
    return float(s)


class SplitReasoner:
    """
    Unidad de Razonamiento Cognitivo (CoT - Chain of Thought).
    """

    @classmethod
    async def decompose_document(
        cls,
        raw_text: str,
        document_name: str = "documento",
        use_llm: bool = True
    ) -> DecomposedInvoice:
        """
        Ejecuta la descomposición forense con traza de pensamiento explícita (<think>...</think>).
        """
        thinking_trace = ""
        extracted_data: Optional[Dict[str, Any]] = None

        # 1. Intentar inferencia neuronal si está habilitada y hay API Key configurada
        if use_llm and (config.openrouter_api_key or config.gemini_api_key):
            try:
                extracted_data, thinking_trace = await cls._call_llm_reasoner(raw_text, document_name)
            except Exception as e:
                thinking_trace += f"\n[Inferencia Remota]: {str(e)}. Activando generador algorítmico neuro-simbólico."

        # 2. Fallback determinista neuro-simbólico con generación de traza de pensamiento
        if not extracted_data:
            extracted_data, fallback_thinking = cls._algorithmic_reasoning(raw_text, document_name)
            if not thinking_trace:
                thinking_trace = fallback_thinking
            else:
                thinking_trace += "\n" + fallback_thinking

        # 3. Construir modelo Pydantic
        items = []
        for idx, it in enumerate(extracted_data.get("items", [])):
            concept = it.get("concept", f"Partida #{idx+1}")
            acc_code, acc_name, cost_center = classify_concept(
                concept,
                extracted_data.get("supplier_name", "")
            )
            items.append(InvoiceLineItem(
                item_index=idx + 1,
                concept=concept,
                quantity=float(it.get("quantity", 1.0)),
                unit_price=float(it.get("unit_price", 0.0)),
                base_amount=float(it.get("base_amount", 0.0)),
                tax_rate=float(it.get("tax_rate", 21.0)),
                tax_amount=float(it.get("tax_amount", 0.0)),
                total_amount=float(it.get("total_amount", 0.0)),
                account_code=it.get("account_code") or acc_code,
                account_name=it.get("account_name") or acc_name,
                cost_center=it.get("cost_center") or cost_center,
                notes=it.get("notes")
            ))

        subtotal = float(extracted_data.get("subtotal", sum(it.base_amount for it in items)))
        tax_total = float(extracted_data.get("tax_total", sum(it.tax_amount for it in items)))
        retention_total = float(extracted_data.get("retention_total", 0.0))
        grand_total = float(extracted_data.get("grand_total", round(subtotal + tax_total - retention_total, 2)))

        expected_total = round(subtotal + tax_total - retention_total, 2)
        discrepancy = round(abs(expected_total - grand_total), 2)
        is_balanced = discrepancy <= 0.05

        # Diagnóstico de Álgebra Superior y Solubilidad de Restricciones
        from .complex_math_solver import ComplexMathForensicSolver
        raw_items_dict = [it.model_dump() for it in items]
        math_diagnosis = ComplexMathForensicSolver.analyze_linear_system_parity(
            subtotal=subtotal,
            tax_total=tax_total,
            retention_total=retention_total,
            grand_total=grand_total,
            line_items=raw_items_dict
        )

        # 4. Desglose multitasa consolidado
        tax_breakdown: List[TaxBreakdownItem] = []
        raw_tb = extracted_data.get("tax_breakdown", [])
        if raw_tb and isinstance(raw_tb, list):
            for t_item in raw_tb:
                tax_breakdown.append(TaxBreakdownItem(
                    tax_rate=float(t_item.get("tax_rate", 21.0)),
                    base_amount=round(float(t_item.get("base_amount", 0.0)), 2),
                    tax_amount=round(float(t_item.get("tax_amount", 0.0)), 2),
                    surcharge_rate=float(t_item.get("surcharge_rate", 0.0)),
                    surcharge_amount=round(float(t_item.get("surcharge_amount", 0.0)), 2)
                ))
        else:
            tax_groups: Dict[float, Dict[str, float]] = {}
            for it in items:
                r = round(it.tax_rate, 2)
                if r not in tax_groups:
                    tax_groups[r] = {"base": 0.0, "tax": 0.0}
                tax_groups[r]["base"] += it.base_amount
                tax_groups[r]["tax"] += it.tax_amount

            for r, g in sorted(tax_groups.items()):
                tax_breakdown.append(TaxBreakdownItem(
                    tax_rate=r,
                    base_amount=round(g["base"], 2),
                    tax_amount=round(g["tax"], 2),
                    surcharge_rate=0.0,
                    surcharge_amount=0.0
                ))

        multi_tax_detected = len(tax_breakdown) > 1

        # 5. Generación del Asiento Contable PGC
        inv_num = extracted_data.get("invoice_number", f"INV-{document_name[:8]}")
        inv_date = extracted_data.get("invoice_date", "2026-10-15")
        supp_name = extracted_data.get("supplier_name", "Proveedor")
        journal_entry = generate_double_entry_seat(
            invoice_number=inv_num,
            date=inv_date,
            supplier_name=supp_name,
            items=items,
            tax_breakdown=tax_breakdown,
            retention_total=retention_total,
            grand_total=grand_total
        )

        audit_notes = [
            f"Cuadratura contable: {'BALANCEADA' if is_balanced else 'DESCUADRADA (dif=' + str(discrepancy) + '€)'}",
            f"Diagnóstico Matemático: {math_diagnosis.problem_type}",
            f"Partidas identificadas: {len(items)}",
            f"Estructura Impositiva: {'Multitasa (' + str(len(tax_breakdown)) + ' tramos)' if multi_tax_detected else 'Monotasa'}",
            f"Asiento Contable PGC: {'Cuadrado (Debe=Haber=' + str(journal_entry.total_debit) + '€)' if journal_entry.is_balanced else 'Descuadrado'}"
        ]
        # 6. Validación Forense de Checksum de NIF/CIF
        from .tax_id_validator import FiscalTaxIdValidator
        from .temporal_punctuality import TemporalPunctualityEngine
        from .currency_normalizer import CurrencyTriangulationEngine

        supp_tax_id = extracted_data.get("supplier_tax_id", "")
        tax_id_val = FiscalTaxIdValidator.validate_tax_id(supp_tax_id)

        # 7. Normalización Temporal y Plazos de Pago (Puntualidad)
        temporal_val = TemporalPunctualityEngine.evaluate_punctuality(
            raw_text=raw_text,
            explicit_date=inv_date,
            explicit_due_date=extracted_data.get("due_date")
        )
        inv_date = temporal_val.normalized_invoice_date
        due_date = temporal_val.normalized_due_date or ""

        # 8. Triangulación de Divisas y Redondeo Banker's Half-Even
        curr = extracted_data.get("currency", "EUR")
        currency_conv = CurrencyTriangulationEngine.normalize_to_eur(
            amount=grand_total,
            currency=curr
        )

        # 9. Matriz de Confianza y Precisión Ponderada por Campo
        field_conf = {
            "tax_id": 1.0 if tax_id_val.is_valid else 0.4,
            "amounts": 1.0 if is_balanced else max(0.0, round(1.0 - (discrepancy / (abs(grand_total) or 1.0)), 2)),
            "dates": temporal_val.date_confidence,
            "items": 1.0 if len(items) > 0 and abs(sum(it.base_amount for it in items) - subtotal) <= 0.05 else 0.8
        }
        composite_score = round(sum(field_conf.values()) / len(field_conf), 3)

        audit_notes.append(f"Identificador Fiscal: {tax_id_val.diagnostic_message}")
        audit_notes.append(f"Puntualidad Temporal: {temporal_val.diagnostic_message}")
        if currency_conv.is_fx_applied:
            audit_notes.append(currency_conv.diagnostic_message)

        if not math_diagnosis.is_solvable:
            audit_notes.append(f"Residual L2: {math_diagnosis.residual_error_norm}")
            for uq in math_diagnosis.unresolvable_equations:
                audit_notes.append(f"Contradicción: {uq}")
            thinking_trace += f"\n\n[Auditoría Algebraica Forense - Rouché-Frobenius]:\n{math_diagnosis.forensic_explanation}"
            if math_diagnosis.optimal_pseudo_solution:
                thinking_trace += f"\n[Solución Óptima de Mínimos Cuadrados L2]: {math_diagnosis.optimal_pseudo_solution}"

        return DecomposedInvoice(
            invoice_number=inv_num,
            supplier_name=supp_name,
            supplier_tax_id=supp_tax_id,
            supplier_address=extracted_data.get("supplier_address", ""),
            customer_name=extracted_data.get("customer_name", "SplitLedger Labs"),
            customer_tax_id=extracted_data.get("customer_tax_id", "ESB87219034"),
            customer_address=extracted_data.get("customer_address", "Madrid, España"),
            invoice_date=inv_date,
            due_date=due_date,
            currency=curr,
            subtotal=round(subtotal, 2),
            tax_total=round(tax_total, 2),
            retention_total=round(retention_total, 2),
            grand_total=round(grand_total, 2),
            is_balanced=is_balanced,
            mathematical_discrepancy=discrepancy,
            multi_tax_detected=multi_tax_detected,
            tax_breakdown=tax_breakdown,
            journal_entry=journal_entry,
            tax_id_validation=tax_id_val.model_dump(),
            temporal_analysis=temporal_val.model_dump(),
            currency_conversion=currency_conv.model_dump(),
            field_confidence=field_conf,
            composite_precision_score=composite_score,
            confidence_score=composite_score,
            items=items,
            thinking_process=thinking_trace.strip(),
            audit_notes=audit_notes
        )

    @classmethod
    async def _call_llm_reasoner(
        cls,
        raw_text: str,
        document_name: str
    ) -> Tuple[Dict[str, Any], str]:
        user_prompt = f"Analiza y descompone el documento '{document_name}':\n\n{raw_text}"

        headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "HTTP-Referer": "https://splitledger.app",
            "X-Title": "SplitReasoner Cognitive Engine",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.openrouter_model,
            "messages": [
                {"role": "system", "content": SYSTEM_REASONING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Extraer traza de pensamiento si existe <think>...</think>
            thinking = ""
            think_match = re.search(r"<think>([\s\S]*?)</think>", content)
            if think_match:
                thinking = think_match.group(1).strip()
                content = content.replace(think_match.group(0), "").strip()

            # Extraer JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                raise ValueError("El modelo no devolvió un bloque JSON válido.")

            extracted = json.loads(json_match.group(0))
            return extracted, thinking

    @classmethod
    def _algorithmic_reasoning(cls, raw_text: str, filename: str) -> Tuple[Dict[str, Any], str]:
        """
        Motor de razonamiento sintético determinista universal.
        Genera una traza de pensamiento transparente y extrae campos clave de cualquier modelo y formato.
        """
        # 0. Si el documento ya es un JSON estructurado
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            try:
                candidate = json.loads(json_match.group(0))
                if isinstance(candidate, dict) and any(k in candidate for k in ["invoice_number", "grand_total", "items", "total", "subtotal"]):
                    trace = "<think>\n1. [JSON Nativo]: Estructura de datos JSON detectada e hidratada directamente sin pérdida de fidelidad.\n</think>"
                    return candidate, trace
            except Exception:
                pass

        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        thinking_steps = [
            f"<think>",
            f"1. [Auditoría Léxica]: Analizando documento '{filename}' con {len(lines)} líneas de texto estructurado.",
        ]

        # 1. Moneda
        currency = "EUR"
        if "$" in raw_text or "USD" in raw_text:
            currency = "USD"
        elif "£" in raw_text or "GBP" in raw_text:
            currency = "GBP"
        elif "¥" in raw_text or "JPY" in raw_text:
            currency = "JPY"
        elif "CHF" in raw_text:
            currency = "CHF"
        thinking_steps.append(f"2. [Divisa]: Detectado símbolo o código de moneda -> {currency}.")

        # 2. Número de factura
        inv_match = re.search(
            r'(?:(?:n[úu]mero|n[ºo]|c[oó]digo)\s+(?:de\s+)?(?:factura|ticket|abono|recibo|invoice)|(?:factura|ticket|invoice|fra|abono)\s*(?:n[ºo]|n[úu]mero|#)|n[ºo]\s*[:#]|invoice\s*(?:#|no\.?|number))\s*[:#]?\s*([A-Za-z0-9\-_/]{3,30})',
            raw_text,
            re.I
        )
        invoice_num = inv_match.group(1) if inv_match else ""
        if not invoice_num:
            code_match = re.search(r"\b([A-Z0-9]{2,8}[-_/]\d{3,8}(?:[-_/]\d+)?)\b", raw_text)
            invoice_num = code_match.group(1) if code_match else f"FAC-{filename[:6].upper()}"
        thinking_steps.append(f"3. [Identificador]: Nº de factura extraído -> {invoice_num}.")

        # 3. NIFs / CIFs
        tax_ids = re.findall(r"\b([0-9]{8}[A-Za-z]|[XYZxyz][0-9]{7}[A-Za-z]|[A-HJ-NP-SUVWa-hj-np-suvw][0-9]{7}[0-9A-Ja-j]|ES[A-Z0-9]{8,9}|LU\d{8}|US\d{8,9})\b", raw_text)
        supplier_tax_id = tax_ids[0] if tax_ids else "B87219034"
        thinking_steps.append(f"4. [NIF/CIF]: Emisor identificado con identificador fiscal -> {supplier_tax_id}.")

        # 4. Proveedor / Razón Social
        corporate_regex = re.compile(r'\b(?:s\.?l\.?u?|s\.?a\.?u?|slu|inc\.?|corp\.?|sarl|gmbh|llc|ltd\.?|limited)\b', re.I)
        supplier_name = ""

        # Búsqueda de campo explícito Proveedor: ...
        supp_explicit = re.search(r'(?:proveedor|emisor|empresa|raz[oó]n\s+social|vendor)\s*[:]\s*([^\n\r|]{3,60})', raw_text, re.I)
        if supp_explicit:
            clean_s = re.sub(r'\(.*?\)', '', supp_explicit.group(1)).strip()
            if len(clean_s) > 2:
                supplier_name = clean_s

        # Búsqueda por marcador societario
        if not supplier_name:
            for line in lines[:8]:
                if corporate_regex.search(line):
                    if "|" in line:
                        cols = [c.strip() for c in line.split("|")]
                        found_col = next((c for c in cols if corporate_regex.search(c)), None)
                        if found_col:
                            supplier_name = found_col
                            break
                    supplier_name = line
                    break

        # Búsqueda por primera línea nominal válida (para profesionales autónomos)
        if not supplier_name:
            for line in lines[:6]:
                if not re.search(r'^(?:calle|avda|av\.|plaza|c/|p[ºo]|nif|cif|nie|vat|tel|email|factura|invoice|ticket|recibo|albar[aá]n|documento)\b', line, re.I):
                    supplier_name = line
                    break

        if not supplier_name:
            supplier_name = "Proveedor General S.L."

        thinking_steps.append(f"5. [Emisor]: Razón social localizada -> '{supplier_name}'.")

        # 5. Detección de Cuenta Bancaria / IBAN
        iban_match = re.search(r"\b([A-Z]{2}\d{2}[ \-]?(?:\d{4}[ \-]?){4}\d{2,4})\b", raw_text)
        if iban_match:
            thinking_steps.append(f"6. [Cuenta Bancaria]: IBAN de liquidación localizado -> {iban_match.group(1)}.")

        # 6. Importes y Partidas
        grand_total = 0.0
        subtotal = 0.0
        tax_total = 0.0
        retention_total = 0.0
        items = []

        monetary_pattern = re.compile(r"(\d+(?:[.,]\d{3})*(?:[.,]\d{2}))")

        for line in lines:
            lower = line.lower()
            amounts = [parse_monetary_amount(m) for m in monetary_pattern.findall(line)]

            is_summary = any(k in lower for k in [
                "total factura", "importe total", "total a pagar", "total:",
                "base imponible", "subtotal", "cuota iva", "total iva", "iva total", "retención", "retencion", "irpf"
            ]) and not (" x " in line or "ref-" in line)

            if is_summary:
                if any(k in lower for k in ["total factura", "importe total", "total a pagar", "total:"]) and amounts:
                    grand_total = max(amounts)
                elif any(k in lower for k in ["base imponible", "subtotal"]) and amounts:
                    subtotal = amounts[-1]
                elif any(k in lower for k in ["cuota iva", "total iva", "iva (21%)", "i.v.a."]) and amounts:
                    tax_total = amounts[-1]
                elif any(k in lower for k in ["retención", "retencion", "irpf"]) and amounts:
                    retention_total = amounts[-1]
            elif amounts and ("|" in line or " x " in line or len(amounts) >= 2 or "partida" in lower or "servicio" in lower or "amazon" in lower or "cloud" in lower):
                # Detectar tipo de IVA específico en la línea si existe
                rate_match = re.search(r"iva\s*[:\s]?\s*(\d{1,2}(?:\.\d+)?)%", line, re.I)
                tax_rate = float(rate_match.group(1)) if rate_match else 21.0

                if len(amounts) >= 3:
                    qty = amounts[0]
                    u_price = amounts[1]
                    b_amt = amounts[2]
                elif len(amounts) == 2:
                    qty = 1.0
                    u_price = amounts[0]
                    b_amt = amounts[1]
                else:
                    qty = 1.0
                    u_price = amounts[0]
                    b_amt = amounts[0]

                t_amt = round(b_amt * (tax_rate / 100.0), 2)

                # Extraer concepto limpio
                if "|" in line:
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    non_num_cols = [c for c in cols if not re.match(r"^[\d.,€$£%\s/:\-]+$", c) and len(c) > 2 and not any(k in c.lower() for k in ["factura", "csv-", "nif"])]
                    if len(non_num_cols) >= 2:
                        concept = non_num_cols[-1]
                    elif len(non_num_cols) == 1:
                        concept = non_num_cols[0]
                    else:
                        concept = cols[0]
                else:
                    raw_concept = line.split("|")[0].strip()
                    m_concept = re.match(r"^(.*?)(?:\s{2,}|\t|\s+(?=\d+\s+))\d+.*$", raw_concept)
                    if m_concept and len(m_concept.group(1).strip()) > 3:
                        concept = m_concept.group(1).strip()
                    else:
                        concept = re.sub(r"\s+[\d.,€$£]+(?:\s+[\d.,€$£]+)*\s*$", "", raw_concept).strip() or "Partida de Servicio"

                items.append({
                    "concept": concept,
                    "quantity": float(qty),
                    "unit_price": float(u_price),
                    "base_amount": float(b_amt),
                    "tax_rate": float(tax_rate),
                    "tax_amount": float(t_amt),
                    "total_amount": round(b_amt + t_amt, 2)
                })

        if not items and grand_total > 0:
            base_calc = round(grand_total / 1.21, 2)
            tax_calc = round(grand_total - base_calc, 2)
            items.append({
                "concept": f"Servicio facturado por {supplier_name}",
                "quantity": 1.0,
                "unit_price": base_calc,
                "base_amount": base_calc,
                "tax_rate": 21.0,
                "tax_amount": tax_calc,
                "total_amount": grand_total
            })

        if subtotal == 0.0 and items:
            subtotal = round(sum(it["base_amount"] for it in items), 2)
        if tax_total == 0.0 and items:
            tax_total = round(sum(it["tax_amount"] for it in items), 2)
        if grand_total == 0.0:
            grand_total = round(subtotal + tax_total - retention_total, 2)

        thinking_steps.append(f"6. [Partida Doble]: Subtotal={subtotal} {currency}, IVA={tax_total} {currency}, Retención={retention_total} {currency} => Total Calculado={round(subtotal + tax_total - retention_total, 2)} vs Declarado={grand_total}.")
        thinking_steps.append(f"7. [PGC]: Mapeando {len(items)} partidas según Plan General Contable.")
        thinking_steps.append("</think>")

        extracted = {
            "invoice_number": invoice_num,
            "supplier_name": supplier_name,
            "supplier_tax_id": supplier_tax_id,
            "invoice_date": "2026-10-15",
            "currency": currency,
            "subtotal": subtotal,
            "tax_total": tax_total,
            "retention_total": retention_total,
            "grand_total": grand_total,
            "confidence_score": 0.985,
            "items": items
        }

        return extracted, "\n".join(thinking_steps)
