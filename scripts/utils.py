import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import expected_weight

def detect_outliers(results, data, group_col, y_col="y", x_cols=["x"], threshold=1.5):
    """
    Detecta outliers baseado nos pesos do modelo robusto
    
    Parameters:
    -----------
    results : dict
        Resultados do modelo
    data : DataFrame
        Dados originais
    group_col : str
        Nome da coluna que identifica os grupos
    threshold : float
        Limiar para identificação de outliers (em desvios padrão abaixo da mediana)
        
    Returns:
    --------
    list
        Lista de grupos identificados como outliers
    """
    groups = data[group_col].unique()
    weights = []
    
    beta = results['beta']
    b = results['b']
    sigma2 = results['sigma2']
    
    for g in groups:
        group_data = data[data[group_col] == g]
        y = group_data[y_col].values
        X = group_data[x_cols].values
        
        # Obter predições do modelo
        mu = results['nl_func'](X, beta, b[g])
        
        # Calcular pesos
        u = expected_weight(y, mu, sigma2*np.eye(len(y)), 
                            results.get('distribution', 'normal'),
                            results.get('df', 4))
        
        weights.append({
            'group': g,
            'weight': np.mean(u)
        })
    
    weights_df = pd.DataFrame(weights)
    
    # Identificar outliers pela regra da mediana +/- threshold * desvio padrão
    median_weight = np.median(weights_df['weight'])
    std_weight = np.std(weights_df['weight'])
    
    outlier_threshold = median_weight - threshold * std_weight
    outliers = weights_df[weights_df['weight'] < outlier_threshold]['group'].tolist()
    
    return outliers

def plot_individual_fits(results, data, nl_func, group_col, x_col, y_col, 
                         n_plots=9, outliers=None):
    """
    Plota ajustes individuais para diferentes grupos
    
    Parameters:
    -----------
    results : dict
        Resultados do modelo
    data : DataFrame
        Dados originais
    nl_func : function
        Função não-linear do modelo
    group_col : str
        Nome da coluna que identifica os grupos
    x_col : str
        Nome da coluna com a variável independente
    y_col : str
        Nome da coluna com a variável dependente
    n_plots : int
        Número de gráficos a serem plotados
    outliers : list
        Lista de grupos identificados como outliers
    """
    groups = data[group_col].unique()
    
    # Selecionar grupos para plotar: todos os outliers + alguns grupos regulares
    if outliers is None:
        outliers = []
    
    regular_groups = [g for g in groups if g not in outliers]
    n_regular = min(n_plots - len(outliers), len(regular_groups))
    
    groups_to_plot = outliers + np.random.choice(regular_groups, n_regular, replace=False).tolist()
    
    # Criar figura
    n_rows = int(np.ceil(len(groups_to_plot) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    
    if n_rows == 1:
        axes = [axes]
    
    axes = axes.flatten()
    
    beta = results['beta']
    b = results['b']
    
    for i, g in enumerate(groups_to_plot):
        if i >= len(axes):
            break
            
        ax = axes[i]
        group_data = data[data[group_col] == g]
        x = group_data[x_col].values
        y = group_data[y_col].values
        
        # Plotar pontos observados
        ax.scatter(x, y, color='blue', alpha=0.7)
        
        # Plotar curva ajustada
        x_grid = np.linspace(np.min(x), np.max(x), 100)
        X_grid = np.column_stack([x_grid] + [np.zeros(100)]*(len(results['x_cols'])-1))
        y_grid = nl_func(X_grid, beta, b[g])
        ax.plot(x_grid, y_grid, 'r-', linewidth=2)
        
        # Adicionar título
        title = f"Grupo {g}"
        if g in outliers:
            title += " (Outlier)"
            ax.set_title(title, color='red')
        else:
            ax.set_title(title)
            
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        
    # Esconder eixos não utilizados
    for i in range(len(groups_to_plot), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()