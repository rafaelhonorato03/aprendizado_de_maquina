import pandas as pd

def treinar_one_rule(caminho_arquivo: str):
    """
    Implementa o algoritmo One-Rule para encontrar o melhor atributo 
    classificador em um conjunto de dados.

    Args:
        caminho_arquivo: O caminho para o arquivo CSV contendo os dados.

    Returns:
        Um dicionário contendo o melhor atributo, sua taxa de erro e suas regras,
        ou None se o arquivo não for encontrado.
    """
    # ==================================================================
    # 1. Carregamento e Configuração
    # ==================================================================
    try:
        # Tenta carregar o arquivo CSV para um DataFrame do pandas
        df = pd.read_csv(caminho_arquivo)
    except FileNotFoundError:
        # Se o arquivo não existir, informa o usuário e encerra a função
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None

    # !! CORREÇÃO APLICADA AQUI !!
    # Define o nome da coluna que queremos prever. Ajustado para minúsculo ('jogar')
    # para corrigir o KeyError anterior.
    coluna_alvo = 'jogar'
    
    # Identifica automaticamente as colunas de atributos (features),
    # ignorando a coluna alvo e a primeira coluna ('Dia'), que é apenas um identificador.
    atributos = [col for col in df.columns if col not in [coluna_alvo, df.columns[0]]]
    total_exemplos = len(df)
    
    print("Base de dados carregada com sucesso:")
    print(df)
    print("\n---------------------------------------------------\n")

    # Lista para armazenar o dicionário de resultados de cada atributo
    resultados_gerais = []

    # ==================================================================
    # 2. Processamento: Análise de cada Atributo
    # ==================================================================
    # Loop principal que itera sobre cada atributo (ex: 'Aspecto', 'Temperatura', etc.)
    for atributo in atributos:
        print(f"Identificando as regras para o atributo '{atributo}'...")
        
        erro_total_atributo = 0
        regras_do_atributo = []
        
        # Encontra todos os valores únicos para o atributo atual (ex: para 'Aspecto', seria ['Sol', 'Nuvens', 'Chuva'])
        valores_unicos = df[atributo].unique()

        # Loop interno para criar uma regra para cada valor único do atributo
        for valor in valores_unicos:
            # Filtra o DataFrame, criando um subconjunto apenas com as linhas
            # que contêm o par atributo-valor atual (ex: todas as linhas onde Aspecto == 'Sol')
            subset = df[df[atributo] == valor]
            
            # Conta as ocorrências de 'Sim' e 'Não' na coluna alvo para esse subconjunto
            contagem_classes = subset[coluna_alvo].value_counts()
            
            # Obtém as contagens de forma segura (retorna 0 se uma classe não existir)
            contagem_sim = contagem_classes.get('Sim', 0)
            contagem_nao = contagem_classes.get('Não', 0)

            # Determina a regra da maioria e calcula o erro (que é a contagem da minoria)
            if contagem_sim >= contagem_nao:
                resultado_regra = 'Sim'
                erro_da_regra = contagem_nao
            else:
                resultado_regra = 'Não'
                erro_da_regra = contagem_sim
            
            # Acumula o erro para o total do atributo
            erro_total_atributo += erro_da_regra
            
            # Formata o texto da regra e o adiciona à lista de regras do atributo
            regra_texto = f"Se {atributo} = '{valor}' então {coluna_alvo} = '{resultado_regra}'"
            regras_do_atributo.append(regra_texto)
            print(f"  - {regra_texto} (Erros: {erro_da_regra})")

        # Calcula a taxa de erro final para o atributo
        taxa_erro = erro_total_atributo / total_exemplos
        print(f"Taxa de Erro total para '{atributo}': {taxa_erro:.4f}\n")
        
        # Armazena os resultados deste atributo em um dicionário e o adiciona à lista geral
        resultados_gerais.append({
            'atributo': atributo,
            'erro_total': erro_total_atributo,
            'taxa_erro': taxa_erro,
            'regras': regras_do_atributo
        })

    # ==================================================================
    # 3. Seleção do Melhor Resultado
    # ==================================================================
    if not resultados_gerais:
        print("Nenhum atributo para analisar.")
        return None
        
    # Encontra o dicionário com o menor 'erro_total' na lista de resultados
    melhor_conjunto = min(resultados_gerais, key=lambda x: x['erro_total'])

    return melhor_conjunto

# ==================================================================
# Execução Principal do Script
# ==================================================================
if __name__ == "__main__":
    # Define o nome do arquivo a ser lido
    arquivo_dados = 'com410-semana-2-one-rule-jogo.csv'
    
    # Chama a função principal para treinar o modelo
    resultado_final = treinar_one_rule(arquivo_dados)
    
    # Se a função retornou um resultado válido, exibe o resumo final
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