import pandas as pd

def treinar_one_rule(caminho_arquivo: str):
    """
    Implementa o algoritmo One-Rule para encontrar o melhor atributo 
    classificador em um conjunto de dados.

    Args:
        caminho_arquivo: O caminho para o arquivo CSV contendo os dados.

    Returns:
        Um dicionário contendo o melhor atributo, sua taxa de erro e suas regras.
    """
    # 1. Carregamento e Configuração
    try:
        df = pd.read_csv(caminho_arquivo)
        print("Nomes das colunas encontradas:", df.columns)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None

    # Identifica a coluna alvo e as colunas de atributos (features)
    # Ignora a primeira coluna ('Dia') que é apenas um identificador
    coluna_alvo = 'jogar'
    atributos = [col for col in df.columns if col not in [coluna_alvo, df.columns[0]]]
    total_exemplos = len(df)
    
    print("Base de dados carregada:")
    print(df)
    print("\n---------------------------------------------------\n")

    resultados_gerais = []

    # 2. Loop principal para analisar cada atributo
    for atributo in atributos:
        print(f"Identificando as regras para o atributo '{atributo}'...")
        
        erro_total_atributo = 0
        regras_do_atributo = []
        
        # Encontra todos os valores únicos para o atributo atual (ex: 'Sol', 'Nuvens')
        valores_unicos = df[atributo].unique()

        # 3. Loop interno para criar uma regra para cada valor único
        for valor in valores_unicos:
            # Filtra o DataFrame para obter apenas as linhas com o par atributo-valor
            subset = df[df[atributo] == valor]
            
            # Conta as ocorrências de 'Sim' e 'Não' nesse subconjunto
            contagem_classes = subset[coluna_alvo].value_counts()
            
            contagem_sim = contagem_classes.get('Sim', 0)
            contagem_nao = contagem_classes.get('Não', 0)

            # 4. Determina a regra da maioria e calcula o erro
            if contagem_sim >= contagem_nao:
                resultado_regra = 'Sim'
                erro_da_regra = contagem_nao
            else:
                resultado_regra = 'Não'
                erro_da_regra = contagem_sim
            
            # Acumula o erro para o atributo
            erro_total_atributo += erro_da_regra
            
            # Formata e armazena a regra
            regra_texto = f"Se {atributo} = '{valor}' então Jogar = '{resultado_regra}'"
            regras_do_atributo.append(regra_texto)
            print(f"  - {regra_texto} (Erros: {erro_da_regra})")

        taxa_erro = erro_total_atributo / total_exemplos
        print(f"Taxa de Erro total para '{atributo}': {taxa_erro:.4f}\n")
        
        # Armazena os resultados deste atributo
        resultados_gerais.append({
            'atributo': atributo,
            'erro_total': erro_total_atributo,
            'taxa_erro': taxa_erro,
            'regras': regras_do_atributo
        })

    # 5. Encontra o melhor conjunto de regras
    if not resultados_gerais:
        print("Nenhum resultado para analisar.")
        return None
        
    melhor_conjunto = min(resultados_gerais, key=lambda x: x['erro_total'])

    return melhor_conjunto

# --- Execução Principal ---
if __name__ == "__main__":
    arquivo_dados = 'com410-semana-2-one-rule-jogo.csv'
    resultado_final = treinar_one_rule(arquivo_dados)
    
    if resultado_final:
        print("---------------------------------------------------")
        print("--- RESULTADO FINAL: MELHOR CONJUNTO DE REGRAS ---")
        print("---------------------------------------------------")
        print(f"Melhor Atributo: '{resultado_final['atributo']}'")
        print(f"Total de Erros: {resultado_final['erro_total']} de {len(pd.read_csv(arquivo_dados))} exemplos.")
        print(f"Menor Taxa de Erro: {resultado_final['taxa_erro']:.4f}")
        print("\nConjunto de Regras Vencedor:")
        for regra in resultado_final['regras']:
            print(f"  -> {regra}")