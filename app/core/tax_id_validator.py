import re
from typing import Dict, Any, Optional
from pydantic import BaseModel


class TaxIdValidationResult(BaseModel):
    is_valid: bool
    tax_id: str
    country_code: str = "ES"
    entity_type: str = "UNKNOWN"  # 'PERSONA_FISICA_DNI', 'EXTRANJERO_NIE', 'SOCIEDAD_CIF', 'INTRACOMUNITARIO_VIES', 'INVALID'
    expected_control: Optional[str] = None
    actual_control: Optional[str] = None
    diagnostic_message: str


class FiscalTaxIdValidator:
    """
    Validador algorítmico forense de identificadores fiscales (NIF, NIE, CIF y VIES intracomunitario).
    Verifica matemáticamente los dígitos de control oficiales de la Agencia Tributaria (AEAT).
    """

    DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
    
    # CIF: Letras que requieren dígito numérico vs carácter alfabético como control
    CIF_NUMERIC_CONTROL = set("ABEH")
    CIF_LETTER_CONTROL = set("KPQS")
    CIF_ANY_CONTROL = set("CDFGJLMNRVW")
    CIF_CONTROL_LETTERS = "JABCDEFGHI"

    @classmethod
    def validate_tax_id(cls, raw_tax_id: str) -> TaxIdValidationResult:
        if not raw_tax_id:
            return TaxIdValidationResult(
                is_valid=False,
                tax_id="",
                diagnostic_message="Identificador fiscal vacío o no detectado."
            )

        clean_id = raw_tax_id.strip().upper().replace("-", "").replace(" ", "").replace(".", "")

        # 1. Comprobar si tiene prefijo de país ES
        if clean_id.startswith("ES") and len(clean_id) > 9:
            clean_id = clean_id[2:]

        # 2. DNI español (8 números + 1 letra)
        if re.match(r"^\d{8}[A-Z]$", clean_id):
            numbers = int(clean_id[:8])
            control_char = clean_id[8]
            expected = cls.DNI_LETTERS[numbers % 23]
            is_valid = (control_char == expected)
            return TaxIdValidationResult(
                is_valid=is_valid,
                tax_id=clean_id,
                country_code="ES",
                entity_type="PERSONA_FISICA_DNI",
                expected_control=expected,
                actual_control=control_char,
                diagnostic_message="DNI español válido (checksum conforme)." if is_valid 
                    else f"DNI inválido: Letra de control errónea (Esperado: '{expected}', Encontrado: '{control_char}')."
            )

        # 3. NIE español (X, Y, Z + 7 números + 1 letra)
        if re.match(r"^[XYZ]\d{7}[A-Z]$", clean_id):
            prefix = clean_id[0]
            prefix_num = {"X": "0", "Y": "1", "Z": "2"}[prefix]
            numbers = int(prefix_num + clean_id[1:8])
            control_char = clean_id[8]
            expected = cls.DNI_LETTERS[numbers % 23]
            is_valid = (control_char == expected)
            return TaxIdValidationResult(
                is_valid=is_valid,
                tax_id=clean_id,
                country_code="ES",
                entity_type="EXTRANJERO_NIE",
                expected_control=expected,
                actual_control=control_char,
                diagnostic_message="NIE extranjero válido (checksum conforme)." if is_valid 
                    else f"NIE inválido: Letra de control errónea (Esperado: '{expected}', Encontrado: '{control_char}')."
            )

        # 4. CIF español (1 letra + 7 números + 1 carácter de control)
        if re.match(r"^[A-HJ-NP-SUVW]\d{7}[0-9A-J]$", clean_id):
            prefix = clean_id[0]
            digits = clean_id[1:8]
            control_char = clean_id[8]

            # Algoritmo de ponderación CIF tipo Luhn
            even_sum = int(digits[1]) + int(digits[3]) + int(digits[5])
            odd_sum = 0
            for idx in [0, 2, 4, 6]:
                doubled = int(digits[idx]) * 2
                odd_sum += (doubled // 10) + (doubled % 10)

            total_sum = even_sum + odd_sum
            digit_control = (10 - (total_sum % 10)) % 10
            letter_control = cls.CIF_CONTROL_LETTERS[digit_control]

            expected_num = str(digit_control)
            expected_let = letter_control

            if prefix in cls.CIF_NUMERIC_CONTROL:
                is_valid = (control_char == expected_num)
                expected_str = expected_num
            elif prefix in cls.CIF_LETTER_CONTROL:
                is_valid = (control_char == expected_let)
                expected_str = expected_let
            else:
                is_valid = (control_char in (expected_num, expected_let))
                expected_str = f"{expected_num} o {expected_let}"

            entity_names = {
                "A": "Sociedad Anónima (S.A.)",
                "B": "Sociedad de Responsabilidad Limitada (S.L.)",
                "C": "Sociedad Colectiva",
                "D": "Sociedad Comanditaria",
                "E": "Comunidad de Bienes",
                "F": "Sociedad Cooperativa",
                "G": "Asociación",
                "H": "Comunidad de Propietarios",
                "J": "Sociedad Civil",
                "N": "Entidad Extranjera",
                "P": "Corporación Local",
                "Q": "Organismo Público",
                "R": "Congregación / Institución Religiosa",
                "S": "Órgano de la Administración",
                "U": "Unión Temporal de Empresas (UTE)",
                "V": "Sociedad Agraria / Otros",
                "W": "Establecimiento Permanente de No Residente"
            }

            ent_desc = entity_names.get(prefix, "Persona Jurídica")
            return TaxIdValidationResult(
                is_valid=is_valid,
                tax_id=clean_id,
                country_code="ES",
                entity_type=f"CIF_{ent_desc}",
                expected_control=expected_str,
                actual_control=control_char,
                diagnostic_message=f"CIF válido ({ent_desc}, control conforme)." if is_valid 
                    else f"CIF inválido para {ent_desc}: Dígito de control erróneo (Esperado: {expected_str}, Encontrado: '{control_char}')."
            )

        # 5. VIES Intracomunitario europeo (Prefijo país 2 letras + código alfanumérico)
        vies_match = re.match(r"^([A-Z]{2})([A-Z0-9]{6,12})$", clean_id)
        if vies_match:
            country = vies_match.group(1)
            return TaxIdValidationResult(
                is_valid=True,
                tax_id=clean_id,
                country_code=country,
                entity_type="INTRACOMUNITARIO_VIES",
                diagnostic_message=f"Operador intracomunitario detectado (País: {country}). Sintaxis VIES válida."
            )

        return TaxIdValidationResult(
            is_valid=False,
            tax_id=clean_id,
            diagnostic_message="Formato fiscal no reconocido según estándares AEAT / VIES."
        )
