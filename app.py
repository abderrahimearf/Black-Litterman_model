import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from portfolio_lib import PortfolioEngine
from datetime import datetime
from uuid import uuid4

st.set_page_config(page_title="Implémentation Fang et al.", layout="wide")
st.title("Étude Comparative : Black-Litterman Fuzzy & Random Views")

# -----------------------
# STATE
# -----------------------
if "views" not in st.session_state:
    st.session_state["views"] = []
if "engine" not in st.session_state:
    st.session_state["engine"] = None

# -----------------------
# SIDEBAR : DONNÉES + SELECTION ACTIFS (COCHER/DÉCOCHER)
# -----------------------
with st.sidebar:
    st.header("1. Univers d'Investissement")

    default_ticks = "AAPL, MSFT, GOOG, AMZN, TSLA, JPM, XOM, GLD"
    raw = st.text_area(
        "Tickers (séparés par virgule)",
        default_ticks,
        help="Ajoute/retire des tickers ici. Ensuite coche/décoche ceux à garder."
    )

    pool = [t for t in raw.upper().replace(" ", "").split(",") if t]

    # Sélection par défaut = tous les tickers du pool
    if "selected_assets" not in st.session_state:
        st.session_state["selected_assets"] = pool

    csa1, csa2 = st.columns(2)
    with csa1:
        if st.button("Tout cocher", use_container_width=True):
            st.session_state["selected_assets"] = pool
    with csa2:
        if st.button("Tout décocher", use_container_width=True):
            st.session_state["selected_assets"] = []

    selected = st.multiselect(
        "Actifs retenus (cocher / décocher)",
        options=pool,
        default=st.session_state["selected_assets"],
    )
    st.session_state["selected_assets"] = selected

    start_d = st.date_input("Début", datetime(2018, 1, 1))
    end_d = st.date_input("Fin", datetime.today())

    c1, c2 = st.columns(2)
    with c1:
        init_btn = st.button("Initialiser Données", use_container_width=True)
    with c2:
        reset_btn = st.button("Reset", use_container_width=True)

    if reset_btn:
        st.session_state["engine"] = None
        st.session_state["views"] = []
        st.session_state.pop("selected_assets", None)
        st.rerun()

    if init_btn:
        if len(selected) < 2:
            st.error("Sélectionne au moins 2 actifs.")
        else:
            eng = PortfolioEngine(selected, start_d, end_d)
            ok, msg = eng.download_data()
            if ok:
                st.session_state["engine"] = eng
                st.success(msg)
            else:
                st.error(msg)

# -----------------------
# MAIN FLOW (VERTICAL, PAS DE COLONNES GAUCHE/DROITE)
# -----------------------
eng = st.session_state["engine"]
if eng is None:
    st.info("👈 Charge d'abord les données via la sidebar (Initialiser Données).")
    st.stop()

st.subheader("2. Définition des Vues (Fuzzy & Random)")
st.markdown("Une vue est un nombre flou aléatoire $\\tilde{Q}^{fr}$.")

# ----- AJOUT DE VUE : HORIZONTAL (sans casser l'écran) -----
with st.container(border=True):
    st.write("### Ajouter une vue")

    # LIGNE 1 : Actif (seul)
    sel_asset = st.selectbox("Actif", eng.tickers, key="v_asset")

    # LIGNE 2 : q, a, b (même ligne)
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_q = st.number_input("q", -1.0, 1.0, 0.10, 0.01, key="v_q")
    with c2:
        sel_a = st.number_input(
            "a (min)", -1.0, sel_q, float(sel_q - 0.05), 0.01, key="v_a"
        )
    with c3:
        sel_b = st.number_input(
            "b (max)", sel_q, 1.0, float(sel_q + 0.05), 0.01, key="v_b"
        )

    # LIGNE 3 : std_a, std_b, std_q (même ligne)
    d1, d2, d3 = st.columns(3)
    with d1:
        sel_std_a = st.number_input("Std a", 0.0, 0.5, 0.02, 0.01, key="v_stda")
    with d2:
        sel_std_b = st.number_input("Std b", 0.0, 0.5, 0.02, 0.01, key="v_stdb")
    with d3:
        sel_std_q = st.number_input("Std q", 0.0, 0.5, 0.01, 0.01, key="v_stdq")

    # LIGNE 4 : bouton (seul, pleine largeur)
    add_view = st.button("➕ Ajouter", use_container_width=True, key="btn_add_view")

    if add_view:
        st.session_state["views"].append({
            "asset": sel_asset,
            "q": float(sel_q),
            "a": float(sel_a),
            "b": float(sel_b),
            "std_a": float(sel_std_a),
            "std_b": float(sel_std_b),
            "std_q": float(sel_std_q),
        })
        st.rerun()

st.write("### 3. Vues ajoutées")

# ----- TABLEAU DES VUES + SUPPRESSION SELECTIVE -----
if not st.session_state["views"]:
    st.info("Aucune vue ajoutée.")
else:
    df_v = pd.DataFrame(st.session_state["views"]).copy()
    df_v["🗑️ Supprimer"] = False

    edited = st.data_editor(
        df_v,
        hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "asset": st.column_config.TextColumn("Actif", disabled=True),
            "q": st.column_config.NumberColumn("q", format="%.3f"),
            "a": st.column_config.NumberColumn("a", format="%.3f"),
            "b": st.column_config.NumberColumn("b", format="%.3f"),
            "std_a": st.column_config.NumberColumn("Std a", format="%.3f"),
            "std_b": st.column_config.NumberColumn("Std b", format="%.3f"),
            "std_q": st.column_config.NumberColumn("Std q", format="%.3f"),
            "🗑️ Supprimer": st.column_config.CheckboxColumn("Supprimer"),
        },
        disabled=["id", "asset"],
        key="views_editor",
    )

    cdel1, cdel2, cdel3 = st.columns([1, 1, 1])
    with cdel1:
        if st.button("🧹 Effacer toutes les vues", use_container_width=True):
            st.session_state["views"] = []
            st.rerun()

    with cdel2:
        if st.button("🗑️ Supprimer les vues cochées", use_container_width=True):
            remaining = edited.loc[edited["🗑️ Supprimer"] == False].drop(columns=["🗑️ Supprimer"])
            st.session_state["views"] = remaining.to_dict("records")
            st.rerun()

    with cdel3:
        st.caption("Conseil : coche puis supprime.")

st.divider()

# -----------------------
# CALCUL (EN BAS, PAS À DROITE)
# -----------------------
st.subheader("4. Lancer le calcul")
run_btn = st.button("▶️ Lancer l'Optimisation", use_container_width=True)

if run_btn:
    with st.spinner("Calcul en cours..."):
        res = {}

        # Benchmarks
        res["MV (Markowitz)"] = eng.run_mean_variance()
        res["Market (Equilibre)"] = eng.run_market_portfolio()

        # Modèles de l'article
        views_payload = st.session_state["views"]  # peut être vide
        res["BL (Standard)"] = eng.run_black_litterman_family(views_payload, "BL")
        res["BL-FV (Fuzzy)"] = eng.run_black_litterman_family(views_payload, "BL-FV")
        res["BL-FRV (Random)"] = eng.run_black_litterman_family(views_payload, "BL-FRV")

        # Comparaison externe
        res["FPMV (Possibiliste)"] = eng.run_fuzzy_possibilistic_mv()

    st.subheader("5. Résultats")

    # --- 1. PERFORMANCE IN-SAMPLE ---
    st.write("#### 📊 Performances (Rentabilité/Risque)")
    perf_list = []
    for m, w in res.items():
        r, v, s = eng.get_portfolio_stats(w)
        perf_list.append({"Modèle": m, "Rentabilité": r, "Volatilité": v, "Sharpe": s})

    df_p = pd.DataFrame(perf_list).set_index("Modèle").sort_values("Sharpe", ascending=False)
    st.dataframe(
        df_p.style
            .format("{:.2%}", subset=["Rentabilité", "Volatilité"])
            .format("{:.4f}", subset=["Sharpe"])
            .background_gradient(cmap="RdYlGn", subset=["Sharpe"])
    )

    # --- 2. ALLOCATION ---
    st.write("#### ⚖️ Poids des Portefeuuilles")
    df_w = pd.DataFrame(res, index=eng.tickers)
    st.dataframe(df_w.style.background_gradient(cmap="Blues", axis=1).format("{:.2%}"))

    # --- 3. RADAR CHART ---
    st.write("#### 🕸️ Radar des allocations")
    fig = go.Figure()
    for col in df_w.columns:
        fig.add_trace(go.Scatterpolar(r=df_w[col], theta=df_w.index, fill="toself", name=col))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Comparaison Visuelle des Allocations")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        **Note Méthodologique (Fang et al.) :**
        - **BL-FV** : $(b-a)^2/24$
        - **BL-FRV** : $(\\sigma_a^2 + \\sigma_b^2 + 2\\sigma_q^2)/6$
        """
    )
