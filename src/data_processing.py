def procesar_valor(valor):
    # Si hay paréntesis, extraemos ambos valores
    if '(' in valor:
        num1 = int(valor.split('(')[0])  # Valor antes de los paréntesis
        num2 = int(valor.split('(')[1].split(')')[0])  # Valor dentro de los paréntesis
        return num1 + num2  # Sumamos ambos valores
    else:
        return int(valor)  # Si no hay paréntesis, solo convertimos a int