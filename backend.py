import numpy as np
import pandas as pd
from scipy.stats import beta


# step 1. defining our payout/probability  distribution and other helper variables

allmultipliers = np.array([2,3,4,5,10,50,100,40000])
allprobs = np.array([
    0.4786385, 0.4462410, 0.0265000, 0.0250000, 0.0235000,
    0.0000800, 0.0000400, 0.0000005])
cap_lvl = 50 #we are ignoring the jackpots above 50x to minimize variance
mask = allmultipliers <= cap_lvl
multipliers = allmultipliers[mask]
probs_beforenormalization=allprobs[mask]

probs = probs_beforenormalization/np.sum(probs_beforenormalization)  #normalization so new probabilities sum up to 1

game_ev = np.sum(multipliers*probs) # we will denote this with 'm'
m=game_ev
breakeven = 1/m  #the needed "winrate" to break even
e_m2 = np.sum(multipliers**2 * probs) #expected value of the squared returns
e_netwin = e_m2-2*m+1  # E[(M-1)^2] = E[M^2] - 2*E[M] + 1
games_perblock=50


#step 2. defining helper functions


def bayesian_p (wins, games, prior_alpha=1, prior_beta=1):
    return beta.rvs (wins+prior_alpha,(games-wins)+prior_beta,size=1)[0]
#this function is a method of bayesian inference that estimates a true player's winrate using the Beta distribution

def bayesian_mean (wins, games, prior_alpha=1, prior_beta=1):
    return (wins+prior_alpha)/(games+prior_alpha+prior_beta)
# for display purposes, this bayesian_mean differs a bit from our calculated winrate, as it is adjusted to
#minimize overconfidence in small samples

def kelly_fractions(stake_win_rates, bankroll, user_kelly_fraction, temperature=0.0):
    fractions_output = {}
    stakes_list = []
    edges = []
    variances = []

    # --- STEP 1: GATHER RAW DATA (Loop through all stakes first) ---
    for stake, data in stake_win_rates.items():
        current_p = data['p']

        # Calculate Edge E[R]
        e_r = current_p * (m - 1) + (1 - current_p) * (-1)

        # Calculate Variance Var(R)
        e_rsquared = current_p * e_netwin + (1 - current_p) * 1
        # Safety check: ensure variance isn't 0 to avoid crash
        var_r = max(1e-9, e_rsquared - (e_r ** 2))

        stakes_list.append(stake)
        edges.append(e_r)
        variances.append(var_r)

    # --- STEP 2: CALCULATE "SMART BUDGET" (Outside the loop) ---
    total_kelly_capacity = 0
    for i, e_r in enumerate(edges):
        if e_r > 0:
            # Sum of rational kelly fractions
            total_kelly_capacity += (e_r / variances[i]) * user_kelly_fraction

            # If completely losing but irrational, force a 1% action floor
    if total_kelly_capacity == 0 and temperature > 0:
        total_kelly_capacity = 0.1

        # --- STEP 3: DISTRIBUTE THE BUDGET (Rational vs Irrational) ---
    if temperature <= 0.001:
        # Rational Mode: Standard Kelly
        final_weights = []
        for i, e_r in enumerate(edges):
            if e_r > 0:
                weight = (e_r / variances[i]) * user_kelly_fraction
            else:
                weight = 0.0
            final_weights.append(weight)
    else:
        # Irrational Mode: Softmax
        # np.exp handles the array math for us
        exp_values = np.exp(np.array(edges) / temperature)

        # Normalize to get probabilities (must sum to 1.0)
        softmax_probs = exp_values / np.sum(exp_values)

        # Leak the "Smart Budget" into bad stakes based on "Desire"
        final_weights = softmax_probs * total_kelly_capacity

    # --- STEP 4: BUILD OUTPUT ---
    for i, stake in enumerate(stakes_list):
        fractions_output[stake] = {
            'p': stake_win_rates[stake]['p'],
            'e_r': edges[i],
            'f_alloc': final_weights[i]
        }

    return fractions_output




def tilt_modifier (game_history, p_avg, Tilt_sens, tilt_window=100):

        actual_tilt_window = min(len(game_history),tilt_window)

        if actual_tilt_window<=10: #10 games or less are not enough to quantify 'tilt'
          return 1.0    #so we return no tilt

        recent_results = game_history[-actual_tilt_window:]
        p_short=np.mean(recent_results)   #computing the 'luck gap'

        delta_p= p_short-p_avg
        if delta_p<0:
            tilt_modifier = 1+delta_p*Tilt_sens   #calculating tilt modifier
            return max(0.1,tilt_modifier)   # tilt mod cannot be less than 0.1.
        else:
            return 1.0   # if we are running good we assume no winner tilt or hot hand fallacy

def run_one_career (user_db, user_params, start_bankroll, nr_games) :
    bankroll=start_bankroll
    bankroll_history=[start_bankroll]
    game_history_outcomes=[]

    Tilt_sens = user_params['Tilt_sens']
    user_kelly_fraction = user_params['user_kelly_fraction']
    temperature = user_params.get('temperature', 0.0)

    career_p = {}
    for stake, data in user_db.items():
        if data['games'] > 0 :
            wins = data['wins']
            games = data['games']                           #computing our winrate for each stake
            career_p[stake]=bayesian_p(wins,games)
        else :
            career_p[stake]=0

    active_p =[p for p in career_p.values() if p>0] # extracting the p's greater than 0
    if active_p :
        avg_p = np.mean(active_p)      #our average winrate across all stakes which will be our base for tilt calculations
    else :
        avg_p=breakeven




    for _  in range (nr_games// games_perblock): #running the simulation in multiple blocks
        if bankroll<=0 :
            bankroll=0
            break

        tilt_mod = tilt_modifier(game_history_outcomes, avg_p,Tilt_sens)
        

        #determine current (tilted) win
        current_winrates={}
        for stake, base_p in career_p.items():
            tilted_p = max(0.0, min(1.0, base_p*tilt_mod))
            current_winrates[stake]={'p' : tilted_p}

        current_fractions = kelly_fractions(current_winrates,bankroll, user_kelly_fraction, temperature)

        # sum up all the positive fractions to find the total "investment"
        total_f_alloc = sum(data['f_alloc'] for data in current_fractions.values() if data['f_alloc'] > 0)



        if total_f_alloc <= 0:
            # no edge at any stake, or broke. we sit out this block.
            bankroll_history.extend([bankroll] * games_perblock)
            continue  # skips to the next 50-game block

        normalized_fractions = {}
        # we normalize all positive fractions to get their proportion
        for stake, data in current_fractions.items():
            if data['f_alloc'] > 0:
                normalized_fractions[stake] = data['f_alloc'] / total_f_alloc


        #now we simulate the 50-game block

        stakes_to_play = []
        for stake, norm_frac in normalized_fractions.items():
            num_games_for_stake = int(round(games_perblock * norm_frac))
            stakes_to_play.extend([stake] * num_games_for_stake)

        # adjust length to be exactly 50 games long (due to rounding)
        if len(stakes_to_play) > games_perblock:
            stakes_to_play = stakes_to_play[:games_perblock]
        elif len(stakes_to_play) < games_perblock and normalized_fractions:
            # add more of the most preferred stake (highest fraction)
            most_preferred_stake = max(normalized_fractions, key=normalized_fractions.get)
            stakes_to_play.extend([most_preferred_stake] * (games_perblock - len(stakes_to_play)))

        np.random.shuffle(stakes_to_play)  # randomize the game order

        for stake in stakes_to_play:
            if bankroll < stake :
                bankroll_history.append(bankroll)
                game_history_outcomes.append(0)
                continue  # skip to the next game

            # simulate one game
            bankroll -= stake
            actual_p_for_outcome = current_winrates[stake]['p']  # Use the tilted p

            if np.random.rand() < actual_p_for_outcome:
                #win
                multiplier = np.random.choice(multipliers, p=probs)
                bankroll += stake * multiplier
                game_history_outcomes.append(1)
            else:
                #lose
                game_history_outcomes.append(0)

            if bankroll < 0: bankroll = 0
            bankroll_history.append(bankroll)  # record bankroll after every game


    while len(bankroll_history) <= nr_games:
        bankroll_history.append(bankroll_history[-1])  # flatline

    return bankroll_history[:nr_games + 1]












