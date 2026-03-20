import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "red":            plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
    "green":          plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
    "blue":           plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
    "yellow":         plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
    "grey":           plt.rcParams['axes.prop_cycle'].by_key()['color'][7],
    "IA":             plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
    "IB":             plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
 }  

def app_slide_efficiency() -> None:
    st.markdown("""
        <style>
            .slide-title {
                color: rgb(0, 70, 112);
                font-size: 2rem !important;
                font-weight: 400;
                font-family: sans-serif !important;
                margin-bottom: 3.5rem;
                margin-top: 2.5rem;   
            }
        </style>
    """, unsafe_allow_html=True)

    st.set_page_config(layout="centered")
    
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">Market simulation</p>', unsafe_allow_html=True)
    
    # ── Sidebar parameters ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4,4,4])
    with col1:
        mu    = st.slider("Risky asset drift, $\mu_s$",        min_value=0.01, max_value=0.30, value=0.10, step=0.01)
        sigma = st.slider("Risky asset volatility, $\sigma_s$",   min_value=0.05, max_value=0.60, value=0.20, step=0.01)
    with col2:
        r0    = st.slider("Risk free rate (initial), $\mu_r(0)$", min_value=0.00, max_value=0.15, value=0.03, step=0.005, format="%.3f")
        dr    = st.slider("Risk free rate adjustment speed, $\Delta \mu_r$", min_value=0.000, max_value=0.10, value=0.000, step=0.001, format="%.3f")
    with col3:
        T     = st.slider("Time horizon, T (years)", min_value=1, max_value=30, value=10)
        seed  = st.number_input("Random seed", min_value=0, max_value=9999, value=43, step=1)

    w0 = 100.0
    N  = 100*T

    # ── Simulation ────────────────────────────────────────────────────────────────
    np.random.seed(int(seed))
    dt = T / N
    t  = np.linspace(0, T, N + 1)
    dW = np.random.normal(0, np.sqrt(dt), N)

    def simulate(adaptive_r):
        W_A    = np.zeros(N + 1)
        W_B    = np.zeros(N + 1)
        l_A    = np.zeros(N + 1)
        l_B    = np.zeros(N + 1)
        g_A    = np.zeros(N + 1)
        g_B    = np.zeros(N + 1)
        r_path = np.zeros(N + 1)
        W_A[0]    = w0
        W_B[0]    = w0
        r_path[0] = r0

        r_cur  = r0
        cum_gA = 0.0
        cum_gB = 0.0

        for k in range(N):
            if adaptive_r and k > 0:
                if g_A_actual > g_B_actual and loan > 0:
                    r_cur = r_cur + dr * dt
                elif g_A_actual > g_B_actual and loan < 0:
                    r_cur = r_cur - dr * dt

            r_path[k] = r_cur

            l_A_target   = (mu - r_cur) / sigma**2
            if l_A_target > 1:
                desired_loan = (l_A_target-1) * W_A[k]
                loan         = min(desired_loan, W_B[k])
            elif l_A_target < 1:
                desired_loan = (l_A_target - 1) * W_A[k]
                loan         = max(desired_loan, -W_A[k])
            else:
                loan = 0

            l_A[k] = (W_A[k] + loan) / W_A[k]
            l_B[k] = (W_B[k] - loan) / W_B[k]

            g_A[k] = r_cur + l_A[k] * (mu - r_cur) - 0.5 * (l_A[k] * sigma)**2
            g_B[k] = r_cur + l_B[k] * (mu - r_cur) - 0.5 * (l_B[k] * sigma)**2

            cum_gA += g_A[k] * dt
            cum_gB += g_B[k] * dt

            mu_A  = r_cur + l_A[k] * (mu - r_cur)
            sig_A = l_A[k] * sigma
            g_A_actual = (mu_A - 0.5 * sig_A**2) * dt + sig_A * dW[k]
            W_A[k+1] = W_A[k] * np.exp(g_A_actual)

            mu_B  = r_cur + l_B[k] * (mu - r_cur)
            sig_B = l_B[k] * sigma
            g_B_actual = (mu_B - 0.5 * sig_B**2) * dt + sig_B * dW[k]
            W_B[k+1] = W_B[k] * np.exp(g_B_actual)

        r_path[-1] = r_cur
        l_A_target   = (mu - r_cur) / sigma**2
        if l_A_target > 1:
            desired_loan = (l_A_target-1) * W_A[k]
            loan         = min(desired_loan, W_B[k])
        elif l_A_target < 1:
            desired_loan = (l_A_target - 1) * W_A[k]
            loan         = max(desired_loan, -W_A[k])
        else:
            loan = 0
        
        l_A[-1] = (W_A[-1] + loan) / W_A[-1]
        l_B[-1] = (W_B[-1] - loan) / W_B[-1]
        g_A[-1] = r_cur + l_A[-1] * (mu - r_cur) - 0.5 * (l_A[-1] * sigma)**2
        g_B[-1] = r_cur + l_B[-1] * (mu - r_cur) - 0.5 * (l_B[-1] * sigma)**2

        cum_gA_arr = np.concatenate([[0], np.cumsum(g_A[:-1] * dt)])
        cum_gB_arr = np.concatenate([[0], np.cumsum(g_B[:-1] * dt)])
        W_avg_A = w0 * np.exp(cum_gA_arr)
        W_avg_B = w0 * np.exp(cum_gB_arr)

        return W_A, W_B, l_A, l_B, W_avg_A, W_avg_B, r_path

    W_A2, W_B2, l_A2, l_B2, Wavg_A2, Wavg_B2, r2 = simulate(adaptive_r=True)

    l_A_target_base = (mu - r0) / sigma**2

    # ── Plots ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9,3.6))
    (ax_w2, ax_l2) = axes

    # Wealth plot
    ax_w2.plot(t, W_A2, label='Investor A', color = COLORS['IA'])
    ax_w2.plot(t, W_B2, label='Investor B', color = COLORS['IB'])
    ax_w2.plot(t, Wavg_A2, linestyle='--', label='A time-avg growth', color = COLORS['IA'])
    ax_w2.plot(t, Wavg_B2, linestyle='--', label='B time-avg growth', color = COLORS['IB'])
    ax_w2.set_yscale('log')
    ax_w2.set_title(rf"Wealth")
    ax_w2.set_xlabel('Time (years)')
    ax_w2.set_ylabel('Wealth')
    ax_w2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9)
    ax_w2.annotate(rf"$\Delta \mu_r = {dr}$",  xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top', color = COLORS["grey"])
    
    #Leverage plot

    ax_l2.plot(t, l_A2, label='ℓ_A', color = COLORS['IA'])
    ax_l2.plot(t, l_B2, label='ℓ_B', color = COLORS['IB'])
    #ax_l2.plot(t, r2, linestyle=':', linewidth=1.5, label='r  (adaptive)')
    ax_l2.axhline(1.0, linewidth=1, linestyle=':', label='ℓ = 1', color= COLORS['grey'])
    ax_l2.annotate(rf"$\Delta \mu_r = {dr}$",  xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top', color = COLORS["grey"])
    ax_l2.set_title(rf'Leverage')
    ax_l2.set_xlabel('Time (years)')
    ax_l2.set_ylabel('ℓ')
    ax_l2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9)

    param_str = f'μ={mu}   σ={sigma}   r₀={r0}   Δr={dr}/yr   W₀={w0}   T={T}yr   seed={seed}'
    
    st.pyplot(fig)
    
    plt.close(fig)


app_slide_efficiency()

st.write(
    "This is how it works:"
)
