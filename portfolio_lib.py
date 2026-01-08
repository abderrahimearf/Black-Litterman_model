import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class PortfolioEngine:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.rf = risk_free_rate
        self.data = None
        self.returns = None
        
    def download_data(self):
        """
        Téléchargement robuste des données via Yahoo Finance.
        Gère les cas 'Adj Close' / 'Close' et les erreurs de tickers.
        """
        try:
            # Téléchargement
            raw = yf.download(self.tickers, start=self.start, end=self.end, progress=False)
            if raw.empty: return False, "Aucune donnée récupérée (Vérifiez votre connexion)."
            
            # Gestion des colonnes (MultiIndex ou Simple)
            if isinstance(raw.columns, pd.MultiIndex):
                try: self.data = raw['Adj Close']
                except KeyError: self.data = raw['Close']
            else:
                self.data = raw[['Adj Close']] if 'Adj Close' in raw.columns else raw[['Close']]
            
            if isinstance(self.data, pd.Series): self.data = self.data.to_frame()
            
            # Nettoyage
            valid_tickers = [t for t in self.tickers if t in self.data.columns]
            self.data = self.data[valid_tickers].ffill().dropna()
            self.tickers = valid_tickers
            self.returns = self.data.pct_change().dropna()
            
            if self.data.empty: return False, "Données vides après nettoyage."
            return True, f"{len(self.data)} jours chargés pour {len(self.tickers)} actifs."
        except Exception as e:
            return False, f"Erreur: {str(e)}"

    # --- MÉTRIQUES ---
    def get_portfolio_stats(self, weights):
        """Calcule Rendement, Volatilité et Sharpe In-Sample"""
        mu = self.returns.mean() * 252
        sigma = self.returns.cov() * 252
        
        port_ret = np.sum(weights * mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        sharpe = (port_ret - self.rf) / port_vol if port_vol > 0 else 0
        return port_ret, port_vol, sharpe

    # --- OUTILS BLACK-LITTERMAN ---
    def get_market_equilibrium(self, cov_mat, delta=2.5):
        """Calcule Pi (Prior)"""
        n = len(self.tickers)
        w_mkt = np.ones(n) / n # Hypothèse équipondérée par défaut
        return delta * np.dot(cov_mat, w_mkt)

    def optimize(self, mu, sigma):
        """Solveur Markowitz (Max Sharpe)"""
        n = len(mu)
        args = (mu, sigma, self.rf)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(n)) # Long Only
        x0 = n * [1./n]

        def neg_sharpe(w, m, s, r):
            ret = np.sum(w * m)
            vol = np.sqrt(np.dot(w.T, np.dot(s, w)))
            if vol < 1e-6: return 0
            return -(ret - r)/vol

        res = minimize(neg_sharpe, x0, args=args, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x

    # =========================================================================
    # IMPLÉMENTATION STRICTE DE L'ARTICLE (FANG ET AL.)
    # =========================================================================
    def run_black_litterman_family(self, views, model_variant='BL', tau=0.05):
        """
        Calcule les poids optimaux selon la variante choisie.
        
        Inputs:
        - views : Liste de dictionnaires contenant {asset, q, a, b, std_a, std_b, std_q}
        - model_variant : 'BL' (Std), 'BL-FV' (Fuzzy), 'BL-FRV' (Fuzzy Random)
        """
        mu_hist = self.returns.mean() * 252
        sigma = self.returns.cov() * 252
        pi = self.get_market_equilibrium(sigma.values)
        
        # Filtrer les vues valides
        valid_views = [v for v in views if v['asset'] in self.tickers]
        k = len(valid_views)
        
        if k == 0: return self.optimize(pi, sigma.values)

        P = np.zeros((k, len(self.tickers)))
        Q = np.zeros(k)
        Omega_diag = []

        for i, v in enumerate(valid_views):
            idx = self.tickers.index(v['asset'])
            P[i, idx] = 1 
            Q[i] = v['q']
            
            # 1. Incertitude Structurelle (Base BL) = tau * p * Sigma * p'
            p_vec = P[i,:]
            base_omega = np.dot(np.dot(p_vec, tau * sigma.values), p_vec.T)
            
            # Paramètres du Triangle Flou
            a = v.get('a', Q[i]-0.05)
            b = v.get('b', Q[i]+0.05)
            
            # 2. Composante Fuzzy (BL-FV)
            # Article Page 5 : Var(A) = (b - a)^2 / 24
            fuzzy_var = ((b - a)**2) / 24.0
            
            # 3. Composante Random (BL-FRV) - Article Page 6
            # Si les bornes a, b et le mode q sont des variables aléatoires.
            # On récupère les écarts-types définis par l'utilisateur
            sig_a = v.get('std_a', 0.0)
            sig_b = v.get('std_b', 0.0)
            sig_q = v.get('std_q', 0.0) # Souvent négligé, mais présent dans la formule complète
            
            # Formule Variance Fuzzy Random (Fang et al., basée sur Kwakernaak)
            # Var(Total) = Var(Fuzzy) + Var(Stochastic)
            # Approximation robuste dérivée de l'article :
            # On ajoute la moyenne des variances stochastiques pondérées
            stochastic_var = (sig_a**2 + sig_b**2 + 2*sig_q**2) / 6.0

            # --- CONSTRUCTION DE LA MATRICE OMEGA ---
            if model_variant == 'BL':
                # Standard : Juste l'incertitude du prior
                Omega_diag.append(base_omega)
                
            elif model_variant == 'BL-FV':
                # Fuzzy Views : Base + Incertitude géométrique du triangle
                Omega_diag.append(base_omega + fuzzy_var)
                
            elif model_variant == 'BL-FRV':
                # Fuzzy Random : Base + Incertitude géométrique + Incertitude stochastique
                # C'est la somme des sources d'incertitude
                Omega_diag.append(base_omega + fuzzy_var + stochastic_var)

        Omega = np.diag(Omega_diag)

        # Formule Maître Black-Litterman
        tau_sigma = tau * sigma.values
        ts_p = np.dot(tau_sigma, P.T)
        pts_p = np.dot(np.dot(P, tau_sigma), P.T)
        
        try:
            mid_term = np.linalg.inv(pts_p + Omega)
        except:
            mid_term = np.linalg.pinv(pts_p + Omega)
            
        mu_bl = pi + np.dot(np.dot(ts_p, mid_term), (Q - np.dot(P, pi)))
        
        return self.optimize(mu_bl, sigma.values)

    # --- AUTRES MODÈLES POUR COMPARAISON ---
    def run_mean_variance(self):
        mu = self.returns.mean() * 252
        sigma = self.returns.cov() * 252
        return self.optimize(mu.values, sigma.values)

    def run_market_portfolio(self):
        n = len(self.tickers)
        return np.ones(n) / n

    def run_fuzzy_possibilistic_mv(self):
        """Méthode Comparative (Possibiliste, basée sur l'historique)"""
        r_min = self.returns.min() * 252
        r_mean = self.returns.mean() * 252
        r_max = self.returns.max() * 252
        
        # Moyenne Possibiliste (Pondération Carlsson & Fullér)
        mu_fuzzy = (r_min + 4 * r_mean + r_max) / 6.0
        
        # Covariance Possibiliste (Dilatation par le spread)
        spread = (r_max - r_min) / 2.0
        adj_factor = spread / spread.mean() 
        sigma_fuzzy = self.returns.cov() * 252
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                sigma_fuzzy.iloc[i, j] *= np.sqrt(adj_factor[i] * adj_factor[j])
        
        return self.optimize(mu_fuzzy.values, sigma_fuzzy.values)