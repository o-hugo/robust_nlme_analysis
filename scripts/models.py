import numpy as np
import pandas as pd
from scipy import stats, linalg
from tqdm import tqdm

def expected_weight(y, mu, sigma, distribution="normal", df=4):
    """
    Calcula E[phi^(-1)(U)|y] para diferentes distribuições
    """
    # Calcular distância de Mahalanobis
    error = y - mu
    if len(error.shape) == 1:
        error = error.reshape(-1, 1)
    
    try:
        inv_sigma = linalg.inv(sigma)
        d = np.sum((error @ inv_sigma) * error, axis=1)
    except:
        inv_sigma = 1/sigma
        d = np.sum(inv_sigma * error**2, axis=0)
    
    n = len(y)
    
    if distribution == "normal":
        return np.ones(n)
    
    elif distribution == "t":
        # Para t-Student (já implementado em scipy, apenas usamos de forma específica)
        return (df + n) / (df + d)
    
    elif distribution == "slash":
        # Para distribuição slash
        a = n/2 + df
        b = d/2
        
        # Usando a função incompleta gamma para P_1
        from scipy.special import gamma, gammainc
        
        P1_num = gammainc(a + 1, b) * gamma(a + 1)
        P1_den = gammainc(a, b) * gamma(a)
        
        return (2*df + n) / (d * P1_num/P1_den)
    
    else:
        raise ValueError("Distribuição não implementada")

def approximate_nlme(data, group_col, nl_func, nl_grad, y_col, x_cols,
                     initial_fixed, initial_random_var, initial_error_var,
                     distribution="normal", df=4, max_iter=100, tol=1e-6):
    """
    Método aproximado para estimação de modelos não-lineares de efeitos mistos
    com distribuições robustas.
    
    Implementa o algoritmo descrito no artigo de Gomes et al. (2022)
    """
    # Extrair grupos únicos
    groups = data[group_col].unique()
    n = len(groups)
    n_fixed = len(initial_fixed)
    n_random = len(initial_random_var)
    
    # Inicializar parâmetros
    beta = np.array(initial_fixed)
    sigma2 = initial_error_var
    D = np.diag(initial_random_var)
    
    # Dicionário para armazenar efeitos aleatórios por grupo
    b = {g: np.zeros(n_random) for g in groups}
    
    # Iteração principal
    converged = False
    iter_count = 0
    beta_prev = beta.copy()
    
    # Para rastrear convergência
    history = {
        'beta': [beta.copy()],
        'sigma2': [sigma2],
        'D': [D.copy()],
        'loglik': []
    }
    
    print("Iniciando otimização pelo método aproximado...")
    for iter_count in tqdm(range(max_iter)):
        # Para armazenar matrizes e vetores por grupo
        Ws = {}
        Ts = {}
        ys_adj = {}
        us = {}
        
        # Passo 1: Linearização em torno das estimativas atuais
        for g in groups:
            # Filtrar dados do grupo
            group_data = data[data[group_col] == g].copy()
            y = group_data[y_col].values
            X = group_data[x_cols].values
            
            # Calcular predições e gradientes
            mu = nl_func(X, beta, b[g])
            W = nl_grad(X, beta, b[g], with_respect_to='beta')
            T = nl_grad(X, beta, b[g], with_respect_to='b')
            
            # Resposta ajustada
            y_adj = y - mu + W @ beta + T @ b[g]
            
            # Calcular pesos (para robustez)
            u = expected_weight(y, mu, sigma2*np.eye(len(y)), distribution, df)
            
            # Armazenar
            Ws[g] = W
            Ts[g] = T
            ys_adj[g] = y_adj
            us[g] = u
            
            # Atualizar efeitos aleatórios (b)
            V_inv = np.diag(u) / sigma2
            Sigma_b_inv = np.linalg.inv(D) + T.T @ V_inv @ T
            Sigma_b = np.linalg.inv(Sigma_b_inv)
            b[g] = Sigma_b @ T.T @ V_inv @ (y_adj - W @ beta)
        
        # Passo 2: Atualizar beta (efeitos fixos)
        numer = np.zeros(n_fixed)
        denom = np.zeros((n_fixed, n_fixed))
        
        for g in groups:
            W = Ws[g]
            y_adj = ys_adj[g]
            u = us[g]
            
            # Usar pesos para robustez
            W_weighted = W * np.sqrt(u).reshape(-1, 1)
            y_adj_weighted = y_adj * np.sqrt(u)
            
            numer += W.T @ np.diag(u) @ y_adj
            denom += W.T @ np.diag(u) @ W
        
        beta_new = np.linalg.solve(denom, numer)
        
        # Passo 3: Atualizar sigma2 (variância do erro)
        sigma2_new = 0
        N = 0
        
        for g in groups:
            W = Ws[g]
            y_adj = ys_adj[g]
            u = us[g]
            T = Ts[g]
            
            resid = y_adj - W @ beta_new - T @ b[g]
            sigma2_new += np.sum(u * resid**2)
            N += len(y_adj)
        
        sigma2_new /= N
        
        # Passo 4: Atualizar D (matriz de covariância dos efeitos aleatórios)
        D_new = np.zeros_like(D)
        for g in groups:
            D_new += np.outer(b[g], b[g])
        D_new /= n
        
        # Verificar convergência
        delta = np.linalg.norm(beta_new - beta_prev) / (1 + np.linalg.norm(beta_prev))
        beta_prev = beta.copy()
        beta = beta_new
        sigma2 = sigma2_new
        D = D_new
        
        # Armazenar histórico
        history['beta'].append(beta.copy())
        history['sigma2'].append(sigma2)
        history['D'].append(D.copy())
        
        if delta < tol:
            converged = True
            break
            
    # Calcular log-verossimilhança final aproximada
    loglik = 0
    for g in groups:
        group_data = data[data[group_col] == g]
        y = group_data[y_col].values
        X = group_data[x_cols].values
        
        mu = nl_func(X, beta, b[g])
        
        if distribution == "normal":
            loglik += stats.multivariate_normal.logpdf(y, mean=mu, cov=sigma2*np.eye(len(y)))
        else:
            u = us[g]
            weighted_sigma = sigma2 / np.mean(u)
            loglik += stats.multivariate_normal.logpdf(y, mean=mu, cov=weighted_sigma*np.eye(len(y)))
    
    history['loglik'] = loglik
    
    return {
        'beta': beta,
        'D': D,
        'sigma2': sigma2,
        'b': b,
        'converged': converged,
        'iterations': iter_count + 1,
        'loglik': loglik,
        'AIC': -2*loglik + 2*(len(beta) + 1 + n_random),
        'BIC': -2*loglik + np.log(N)*(len(beta) + 1 + n_random),
        'history': history
    }