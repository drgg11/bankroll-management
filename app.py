import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# --- Import from your backend.py (formerly engine.py) ---
# Ensure your backend.py file is in the same folder
from backend import (
    bayesian_mean,
    kelly_fractions,
    run_one_career,
    breakeven,
    # If GAMES_PER_BLOCK is not exported, we default to 100 in the UI text
)

st.set_page_config(layout="wide", page_title="Behavioral Bankroll & Risk Simulator")

# --- Custom CSS for Dark Mode aesthetics ---
st.markdown("""
<style>
.main { background-color: #0E1117; color: #FAFAFA; }
.stApp { background-color: #0E1117; }
.sidebar .sidebar-content { background-color: #161A25; }
h1, h2, h3, h4, h5, h6 { color: #79C7C5; }
/* Make metrics stand out */
[data-testid="stMetricValue"] { font-size: 1.8rem; color: #79C7C5; }
[data-testid="stMetricLabel"] { color: #A0AEC0; }
/* Table styling */
.stDataFrame { background-color: #161A25; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("Behavioral Bankroll & Risk Simulator 📈")

# --- 1. SIDEBAR (Inputs) ---
with st.sidebar:
    st.header("Settings")

    st.subheader("Simulation Parameters")
    start_bankroll = st.number_input("Starting Bankroll ($)", value=1000, min_value=10, step=100)
    nr_games = st.number_input("Total Games to Simulate", value=50000, min_value=500, step=10000)
    num_careers = st.number_input("Number of Career Paths", value=100, min_value=10, step=10)

    # --- IRRATIONALITY SLIDER ---
    st.markdown("---")
    st.markdown("**🧠 Player Rationality**")
    irrationality_score = st.slider(
        "Irrationality (%)",
        0, 100, 0,
        help="0% = Perfect Robot. Higher values leak money to bad bets via Softmax exploration."
    )
    # Adjusted scaling: /50 was too random (max temp 2.0).
    # /200 gives max temp 0.5, which is sufficiently irrational without being RNG.
    temperature = irrationality_score / 200.0

    st.subheader("Your Game History")

    # 1. Define fallback data (List of Dicts)
    raw_data = [
        {'stake': 0.25, 'games': 121, 'wins': 44},
        {'stake': 1.0, 'games': 166, 'wins': 58},
        {'stake': 3.0, 'games': 27, 'wins': 11}
    ]

    # 2. Initialize DataFrame container
    df_to_edit = pd.DataFrame(raw_data)

    # 3. Try to load data.csv (Overwrite df_to_edit if successful)
    if os.path.exists('data.csv'):
        try:
            loaded_df = pd.read_csv('data.csv')
            if {'stake', 'games', 'wins'}.issubset(loaded_df.columns):
                df_to_edit = loaded_df
        except:
            pass  # If load fails, we silently keep the raw_data DataFrame

    # 4. The Data Editor
    edited_df = st.data_editor(
        df_to_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "stake": st.column_config.NumberColumn("Stake ($)", format="%.2f"),
            "games": st.column_config.NumberColumn("Games", format="%d"),
            "wins": st.column_config.NumberColumn("Wins", format="%d"),
        }
    )

    # 5. Process the data
    user_data = {}

    if not edited_df.empty:
        try:
            # Force numeric types to prevent string math errors
            edited_df = edited_df.astype({'stake': float, 'games': int, 'wins': int})
            # Logic check: Wins cannot exceed games
            edited_df['wins'] = edited_df.apply(lambda row: min(row['wins'], row['games']), axis=1)

            # Parse into dictionary for the backend
            user_data = {
                row['stake']: {'games': row['games'], 'wins': row['wins']}
                for _, row in edited_df.iterrows()
                if row['games'] > 0
            }
        except Exception as e:
            st.error(f"Error processing data table: {e}")

# --- 2. MAIN TABS ---
# Added Tab 4 for Documentation
tab1, tab2, tab3, tab4 = st.tabs(["1. Profile", "2. Rational Plan", "3. Simulation", "4. Information"])

# Prepare params dictionary
user_params = {}

with tab1:
    st.header("Risk & Tilt Profile")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Appetite")
        q1 = st.radio("Strategy Style", ("Conservative", "Balanced", "Aggressive"), index=1)

    with col2:
        st.subheader("Tilt Sensitivity")
        q2 = st.radio("Reaction to Downswings", ("Robot (None)", "Zen (Low)", "Frustrated (Med)", "Furious (High)"),
                      index=1)

    if "Conservative" in q1:
        k_frac = 0.1
    elif "Balanced" in q1:
        k_frac = 0.25
    else:
        k_frac = 0.5

    if "Robot" in q2:
        t_s = 0.0
    elif "Zen" in q2:
        t_s = 0.25
    elif "Frustrated" in q2:
        t_s = 1.0
    else:
        t_s = 3.0

    user_params = {
        'user_kelly_fraction': k_frac,
        'Tilt_sens': t_s,
        'temperature': temperature
    }

    st.success(
        f"Profile Loaded! **Risk Appetite:** {q1} | **Tilt Sensitivity:** {q2} | **Irrationality:** {irrationality_score}%")

with tab2:
    st.header("Strategy Preview")

    # Toggle for Rational vs Irrational View
    show_irrational = st.checkbox("Show my 'Irrational' plan (incorporating my slider settings)", value=True)

    if show_irrational:
        display_temp = temperature
        if irrationality_score == 0:
            st.info("With **0% Irrationality**, this plan is identical to the Rational/Kelly plan.")
        else:
            st.warning(
                f"This shows your allocation with **{irrationality_score}% Irrationality**. Money will 'leak' into negative stakes.")
    else:
        display_temp = 0.0
        st.success("This shows the **Mathematically Perfect (Rational)** allocation. Negative stakes are 0%.")

    if user_data:
        # Calculate 'A-Game' win rates using Bayesian mean for display
        a_game_win_rates_for_display = {}
        for stake, data in user_data.items():
            if data['games'] > 0:
                wins = min(data['wins'], data['games'])
                games = data['games']
                a_game_win_rates_for_display[stake] = {'p': bayesian_mean(wins, games)}
            else:
                a_game_win_rates_for_display[stake] = {'p': 0.0}

        # Call backend to get fractions
        a_game_fractions = kelly_fractions(
            a_game_win_rates_for_display,
            start_bankroll,
            k_frac,
            temperature=display_temp
        )

        display_data = []
        for stake, data in a_game_fractions.items():
            estimated_tables = 0
            numeric_stake = float(stake)
            f_alloc_percent = data['f_alloc']

            if numeric_stake > 0 and f_alloc_percent > 0:
                estimated_tables = int(round((f_alloc_percent * start_bankroll) / numeric_stake))

            display_data.append({
                "Stake": f"${int(numeric_stake)}" if numeric_stake.is_integer() else f"${numeric_stake:.2f}",
                "Est. Win Rate": f"{data['p']:.2%}",
                "Edge (ROI)": f"{data['e_r']:.2%}",
                "Allocation %": f"{f_alloc_percent:.2%}",
                "Est. Tables": estimated_tables
            })

        st.dataframe(pd.DataFrame(display_data).sort_values(by="Stake"), use_container_width=True, hide_index=True)

with tab3:
    st.header("Simulation Results")

    if st.button("🚀 Run Simulation"):
        if not user_data:
            st.error("Please enter game data in the sidebar.")
        else:
            with st.spinner("Simulating careers..."):

                all_paths = []
                tilt_free_paths = []

                bar = st.progress(0)

                for i in range(num_careers):
                    # 1. Sim WITH User's Tilt and Irrationality
                    all_paths.append(run_one_career(user_data, user_params, start_bankroll, nr_games))

                    # 2. Sim WITHOUT Tilt (Perfect Control)
                    # We keep irrationality (temperature) to isolate just the cost of TILT.
                    tf_params = user_params.copy()
                    tf_params['Tilt_sens'] = 0.0
                    tilt_free_paths.append(run_one_career(user_data, tf_params, start_bankroll, nr_games))

                    bar.progress((i + 1) / num_careers)

                bar.empty()

                # --- STATISTICS ---
                final_br = [p[-1] for p in all_paths]
                final_br_tf = [p[-1] for p in tilt_free_paths]

                # Risk of Ruin
                ruin_count = sum(1 for x in final_br if x <= 0)
                risk_of_ruin = (ruin_count / num_careers) * 100

                ruin_count_tf = sum(1 for x in final_br_tf if x <= 0)
                risk_of_ruin_tf = (ruin_count_tf / num_careers) * 100

                # Percentiles
                median_br = np.median(final_br)
                p75_br = np.percentile(final_br, 75)
                p25_br = np.percentile(final_br, 25)

                median_br_tf = np.median(final_br_tf)
                p75_br_tf = np.percentile(final_br_tf, 75)
                p25_br_tf = np.percentile(final_br_tf, 25)

                # --- PLOTTING ---
                fig, ax = plt.subplots(figsize=(10, 5))

                step = max(1, nr_games // 500)

                # 1. Draw Green FIRST (Background)
                for p in tilt_free_paths[:50]:
                    ax.plot(p[::step], color='green', alpha=0.4, linewidth=1.2)

                # 2. Draw Red SECOND (Foreground - so you can see crashes)
                for p in all_paths[:50]:
                    ax.plot(p[::step], color='red', alpha=0.4, linewidth=1.2)

                # Averages
                # Handle varying lengths if engine returns different sizes (padding logic should prevent this)
                max_len = max(len(p) for p in all_paths)
                all_paths_padded = [p + [p[-1]] * (max_len - len(p)) for p in all_paths]
                tilt_free_padded = [p + [p[-1]] * (max_len - len(p)) for p in tilt_free_paths]

                avg_path = np.mean(all_paths_padded, axis=0)
                avg_tf_path = np.mean(tilt_free_padded, axis=0)

                ax.plot(avg_path[::step], color='red', linewidth=2.5, linestyle='--', label=f"You (Tilt={t_s})")
                ax.plot(avg_tf_path[::step], color='green', linewidth=2.5, linestyle='--', label=f"Tilt-Free")

                ax.set_title("Projected Bankroll Growth")
                ax.set_xlabel("Games Played")
                ax.set_ylabel("Bankroll ($)")
                ax.legend()

                # Dark Mode Styling for Matplotlib
                ax.grid(True, alpha=0.2, color='#A0AEC0')
                ax.set_facecolor('#0E1117')
                fig.patch.set_facecolor('#0E1117')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.title.set_color('white')
                plt.setp(ax.get_legend().get_texts(), color='black')  # Legend text

                st.pyplot(fig)

                # --- METRICS DISPLAY ---
                st.subheader("Detailed Statistics")

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("### 🔴 Your Results (Tilted)")
                    st.metric("Median Bankroll", f"${median_br:,.0f}")
                    st.metric("Risk of Ruin", f"{risk_of_ruin:.1f}%")
                    st.caption(f"Range (25th - 75th): ${p25_br:,.0f} — ${p75_br:,.0f}")

                with c2:
                    st.markdown("### 🟢 Tilt-Free Potential")
                    st.metric("Median Bankroll", f"${median_br_tf:,.0f}")
                    st.metric("Risk of Ruin", f"{risk_of_ruin_tf:.1f}%")
                    st.caption(f"Range (25th - 75th): ${p25_br_tf:,.0f} — ${p75_br_tf:,.0f}")

                st.divider()

                # Cost of Tilt Analysis
                diff = median_br_tf - median_br
                if diff > 0:
                    st.error(
                        f" **Cost of Tilt:** Your emotional response is costing you approximately **${diff:,.0f}** in expected median value.")
                elif diff < -100:
                    st.success(
                        " **Aggressive Edge:** Interestingly, your tilt settings resulted in a higher median. This usually implies a high-risk 'gamble' strategy paying off in the median, but check your Risk of Ruin.")
                else:
                    st.success(
                        " **Mental Game Strong:** Your tilt is not significantly harming your median performance.")

with tab4:
    st.header("ℹ️ Project Details & Parameters")

    st.markdown("""
    ### How Parameters Affect Your Outcome

    This simulator models the intersection of **Skill**, **Risk Management**, and **Psychology**. It can handle anything from a short 500-game "shot" to a 50,000+ game career grind. Here is how your inputs change the math:

    #### 1. Risk Appetite (The Gas Pedal)
    This controls the **Kelly Fraction** used in sizing bets.
    * **Conservative (0.1):** Very safe. Growth is slow, but bankruptcy is extremely rare. Recommended for preserving capital.
    * **Balanced (0.25):** The industry standard "Quarter Kelly." A good balance of aggressive growth and variance control.
    * **High Risk (0.5):** "Half Kelly." Extremely volatile. You will see massive upswings but significantly higher risk of ruin.

    #### 2. Tilt Sensitivity (The Penalty)
    This controls how much your **Win Rate** drops during a downswing.
    * **Robot (0.0):** No penalty. You play your A-Game even after losing 10 buy-ins.
    * **Zen/Frustrated (0.25 - 1.0):** Moderate penalty. A bad run makes you play slightly worse (e.g., win rate drops from 39% to 37%), which shrinks your edge.
    * **Furious (3.0):** Severe penalty. A downswing turns you into a losing player, often causing a "death spiral."

    #### 3. Irrationality / Action Bias (The Leak)
    This controls the **Boltzmann Temperature** of your decision-making.
    * **0%:** You are a rational actor. You never bet on negative-EV stakes. If all stakes are bad, you sit out.
    * **>0%:** You suffer from "Action Bias." You feel a need to play even when the math says fold.
    * **Effect:** This forces the simulation to "leak" capital into negative-EV games. The higher the slider, the more money is diverted from good investments to bad ones, dragging down your long-term expected value.

    ---

    ### 🧠 Deep Dive: The Mathematical Engine

    #### 1. Why the Kelly Criterion Works (The Derivation)
    The **Kelly Criterion** isn't just a gambling rule; it's a formula from Information Theory derived to maximize the **Geometric Growth Rate** of your wealth.

    If your bankroll grows by a factor of $(1 + fR)$ each game (where $f$ is your bet size and $R$ is your return), your wealth after $N$ games is:
    $$W_N = W_0 \prod_{i=1}^N (1 + f R_i)$$

    To maximize this, we maximize the **logarithm** of wealth (Log-Utility), which turns the product into a sum:
    $$E[\log(W_N)] \approx E[\log(1 + f R)]$$

    Using the **Taylor Series approximation** $\log(1+x) \approx x - \\frac{x^2}{2}$, we get:
    $$g(f) \approx f E[R] - \\frac{f^2}{2} E[R^2]$$

    To find the peak of this curve (maximum growth), we take the derivative with respect to $f$ and set it to 0:
    $$g'(f) = E[R] - f E[R^2] = 0 \implies f^* = \\frac{E[R]}{E[R^2]}$$

    Since $E[R^2] \\approx Var(R)$ for small edges, we get the formula used in this engine:
    **$$f^* = \\frac{\\text{Edge}}{\\text{Variance}}$$**

    #### 2. The 'Winner-Take-All' Assumption (Conservative Modeling)
    In Spin & Gold, multipliers like 10x or 25x often split the prize (e.g., 1st gets 80%, 2nd gets 20%). 
    **This engine deliberately ignores the 2nd place prize.** We treat every game as "Winner-Take-All."

    **Why?**
    * **Variance Safety:** Splitting prizes *lowers* variance (it smooths the equity curve). By modeling it as Winner-Take-All, we artificially *inflate* the variance in the model.
    * **The Result:** Higher variance -> Lower Kelly Fraction ($f^*$).
    * **Conclusion:** This makes the model **conservative**. It effectively tells you to bet *slightly less* than you theoretically could. In risk management, "under-betting" is safe (you grow slower), while "over-betting" is fatal (you go broke). We choose safety.

    #### 3. Calculating Variance & EV: The "Physics" of the Game
    The Spin & Gold payout structure is the "physics engine" of our simulation. We don't guess these numbers; we derive them from the prize table.

    We use a **Practical Capped Distribution** ($M \le 50x$) to prevent black swan events (like the 40,000x jackpot) from distorting day-to-day strategy.

    * **Multipliers ($M$):** `[2, 3, 4, 5, 10, 50]`
    * **Expected Value ($E[M]$):** The weighted average of all multipliers based on their probabilities. This is approximately **2.766**.
        * If your Win Rate ($p$) satisfies $p \times E[M] > 1$, you are profitable.
    * **Expected Net Win Squared ($E[(M-1)^2]$):** This is the key component for Variance. It measures the "spread" of outcomes.
        * Formula: $E[M^2] - 2E[M] + 1$
        * This variable is pre-calculated in the backend as `E_NET_WIN_SQUARED_PRACTICAL`. It tells the Kelly formula how "bumpy" the ride will be.

    #### 4. Modeling Irrationality (Boltzmann Exploration)
    How do we model a human who knows the odds but gambles anyway? We borrow from **Thermodynamics** and **Machine Learning**.

    We use the **Softmax Function** (Boltzmann distribution):
    $$P(\\text{Stake}_i) = \\frac{e^{\\text{Edge}_i / \\tau}}{\\sum e^{\\text{Edge}_j / \\tau}}$$

    * **$\\tau$ (Temperature):** This is your "Irrationality Slider."
    * **If $\\tau \\to 0$:** The term explodes for the best stake and vanishes for the others. You act like a robot.
    * **If $\\tau \\to \\infty$:** The terms become equal. You bet randomly on everything.

    This effectively models "leakage"—money flowing from your optimal bets into your sub-optimal ones due to boredom or tilt.

    ---

    ### ⚠️ Weaknesses & Limitations

    1.  **No Rakeback (Fish Buffet):** GGPoker has a massive cashback system. This model ignores it. Since rakeback is pure profit, this model is **pessimistic**. You are likely profitable at a lower win rate than this model suggests.
    2.  **Static Skill:** In reality, you learn (or rust) over 50,000 games. This model assumes your "A-Game" skill is constant, only affected temporarily by tilt.
    3.  **No "Life Money":** The model assumes you reinvest 100% of profits. It does not account for monthly withdrawals for rent/food, which drastically increases the risk of ruin.
    """)