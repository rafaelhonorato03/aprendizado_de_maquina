def criar_motor_de_inferencia():
    """
    Cria e retorna a função do motor de inferência, já configurada com as regras.
    Isso encapsula a lógica e a torna reutilizável.
    """
    
    # 1. BASE DE CONHECIMENTO (AS REGRAS)
    # Cada função recebe a base de fatos e a modifica se a regra for aplicável.
    
    regras = [
        # Regras de Classificação de Cadastro
        lambda fatos: fatos.update({'Cadastro': 'Regular'}) if fatos.get('Renda') == 'Alta' and fatos.get('Despesa') == 'Alta' and 'Cadastro' not in fatos else None,
        lambda fatos: fatos.update({'Cadastro': 'Bom'}) if fatos.get('Renda') == 'Alta' and fatos.get('Despesa') == 'Baixa' and 'Cadastro' not in fatos else None,
        lambda fatos: fatos.update({'Cadastro': 'Ruim'}) if fatos.get('Renda') == 'Baixa' and fatos.get('Despesa') == 'Alta' and 'Cadastro' not in fatos else None,
        lambda fatos: fatos.update({'Cadastro': 'Regular'}) if fatos.get('Renda') == 'Baixa' and fatos.get('Despesa') == 'Baixa' and 'Cadastro' not in fatos else None,

        # Regras de Classificação de Operação
        lambda fatos: fatos.update({'Operacao': 'Boa'}) if fatos.get('nPrest') == 'Baixo' and fatos.get('valPrest') == 'Baixo' and 'Operacao' not in fatos else None,
        lambda fatos: fatos.update({'Operacao': 'Boa'}) if fatos.get('nPrest') == 'Alto' and fatos.get('valPrest') == 'Baixo' and 'Operacao' not in fatos else None,
        lambda fatos: fatos.update({'Operacao': 'Boa'}) if fatos.get('nPrest') == 'Baixo' and fatos.get('valPrest') == 'Alto' and 'Operacao' not in fatos else None,
        lambda fatos: fatos.update({'Operacao': 'Ruim'}) if fatos.get('nPrest') == 'Alto' and fatos.get('valPrest') == 'Alto' and 'Operacao' not in fatos else None,

        # Regras de Conversão (Variáveis numéricas para categóricas)
        lambda fatos: fatos.update({'Renda': 'Alta'}) if fatos.get('valRenda', 0) >= 10000 and 'Renda' not in fatos else None,
        lambda fatos: fatos.update({'Renda': 'Baixa'}) if fatos.get('valRenda', 0) < 10000 and 'Renda' not in fatos else None,
        lambda fatos: fatos.update({'Despesa': 'Alta'}) if fatos.get('valDespesa', 0) >= 6000 and 'Despesa' not in fatos else None,
        lambda fatos: fatos.update({'Despesa': 'Baixa'}) if fatos.get('valDespesa', 0) < 6000 and 'Despesa' not in fatos else None,
        lambda fatos: fatos.update({'nPrest': 'Alto'}) if fatos.get('Qtde_Prest', 0) >= 12 and 'nPrest' not in fatos else None,
        lambda fatos: fatos.update({'nPrest': 'Baixo'}) if fatos.get('Qtde_Prest', 0) < 12 and 'nPrest' not in fatos else None,
        lambda fatos: fatos.update({'valPrest': 'Alto'}) if fatos.get('Prestacao', 0) >= 1000 and 'valPrest' not in fatos else None,
        lambda fatos: fatos.update({'valPrest': 'Baixo'}) if fatos.get('Prestacao', 0) < 1000 and 'valPrest' not in fatos else None,

        # Regras de Decisão Final (Objetivo)
        lambda fatos: fatos.update({'Credito': 'Nao'}) if fatos.get('Cadastro') == 'Ruim' and 'Credito' not in fatos else None,
        lambda fatos: fatos.update({'Credito': 'Nao'}) if fatos.get('Operacao') == 'Ruim' and 'Credito' not in fatos else None,
        lambda fatos: fatos.update({'Credito': 'Sim'}) if fatos.get('Cadastro') == 'Bom' and fatos.get('Operacao') == 'Boa' and 'Credito' not in fatos else None,
        lambda fatos: fatos.update({'Credito': 'Analista'}) if fatos.get('Cadastro') == 'Regular' and fatos.get('Operacao') == 'Boa' and 'Credito' not in fatos else None,
    ]

    # 2. O MOTOR DE INFERÊNCIA
    def executar_motor(fatos_iniciais, objetivo, verbose=True):
        """
        Executa o processo de encadeamento para a frente.
        
        Args:
            fatos_iniciais (dict): Dicionário com os valores de entrada.
            objetivo (str): A variável que queremos descobrir.
            verbose (bool): Se True, imprime o passo a passo da inferência.
        """
        fatos = dict(fatos_iniciais)
        if verbose:
            print("--- INICIANDO ANÁLISE DE CRÉDITO ---")
            print(f"Fatos Iniciais: {fatos}")
            print(f"Objetivo: Encontrar o valor de '{objetivo}'\n")

        while True:
            tamanho_fatos_antes = len(fatos)
            
            # Itera sobre todas as regras na base de conhecimento
            for i, regra in enumerate(regras):
                regra(fatos)

            tamanho_fatos_depois = len(fatos)

            # Se nenhuma regra nova foi disparada nesta iteração, paramos.
            if tamanho_fatos_antes == tamanho_fatos_depois:
                if verbose:
                    print("--- Nenhuma regra nova pôde ser aplicada. Fim da análise. ---\n")
                break
        
        if verbose:
            print("--- RESULTADO FINAL ---")
            print(f"Base de Fatos Final: {fatos}")
        
        resultado = fatos.get(objetivo, "não foi possível determinar")
        print(f"Resultado para o objetivo '{objetivo}': {resultado}\n")
        return resultado

    return executar_motor

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Cria uma instância do nosso motor de inferência
    analisar_credito = criar_motor_de_inferencia()

    # --- Situação 1 ---
    fatos_1 = {
        'valRenda': 3000,
        'valDespesa': 6000,
        'Qtde_Prest': 6,
        'Prestacao': 500
    }
    analisar_credito(fatos_1, 'Credito')

    # --- Situação 2 ---
    fatos_2 = {
        'valRenda': 11000,
        'valDespesa': 4000,
        'Qtde_Prest': 6,
        'Prestacao': 800
    }
    analisar_credito(fatos_2, 'Credito')
