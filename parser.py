import os
import re
import pandas as pd

# --- CONFIGURATION ---
# 1. Put your unzipped hand history folder inside your PyCharm project folder.
# 2. Change this variable to match the folder's name.
HAND_HISTORY_FOLDER = "pokerhands"
# ---

# We will store our counted data in a dictionary
# e.g., {0.25: {'games': 0, 'wins': 0}, 1.0: {'games': 0, 'wins': 0}}
stake_data = {}

# Compile our regex patterns for speed.
# We are looking for these exact lines:
# 1. "3 Players"
# 2. "Buy-in: $X.XX"
# 3. "You finished in 1st place"
player_count_re = re.compile(r"(\d+) Players")
buy_in_re = re.compile(r"Buy-in: \$([\d\.]+)")
win_re = re.compile(r"You finished in 1st place")

print(f"Starting parser in folder: {HAND_HISTORY_FOLDER}...")

# Check if the folder exists
if not os.path.isdir(HAND_HISTORY_FOLDER):
    print(f"Error: Folder '{HAND_HISTORY_FOLDER}' not found.")
    print("Please unzip your files into a folder with that name,")
    print("or change the HAND_HISTORY_FOLDER variable in this script.")
else:
    file_count = 0
    valid_games_found = 0

    # Loop through every single file in that folder
    for filename in os.listdir(HAND_HISTORY_FOLDER):
        if filename.endswith(".txt"):
            file_count += 1
            filepath = os.path.join(HAND_HISTORY_FOLDER, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # --- LOGIC ---

                    # 1. IS IT A 3-PLAYER GAME?
                    player_match = player_count_re.search(content)
                    if not player_match:
                        continue  # Skip file, not a game summary we recognize

                    if int(player_match.group(1)) != 3:
                        # It's a 6-max or other, skip it
                        continue

                    # 2. IF YES, FIND THE STAKE
                    buy_in_match = buy_in_re.search(content)
                    if not buy_in_match:
                        continue  # Skip file, couldn't find buy-in

                    stake = float(buy_in_match.group(1))

                    # We have found a valid 3-player game.
                    valid_games_found += 1

                    # 3. ADD GAME TO OUR DATA
                    if stake not in stake_data:
                        stake_data[stake] = {'games': 0, 'wins': 0}

                    stake_data[stake]['games'] += 1

                    # 4. DID WE WIN?
                    if win_re.search(content):
                        stake_data[stake]['wins'] += 1

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"\n--- Parser Finished ---")
    print(f"Total files read: {file_count}")
    print(f"Valid 3-player games found: {valid_games_found}")

    # 5. WRITE THE FINAL CSV FILE
    if valid_games_found > 0:
        data_list = []
        for stake, data in stake_data.items():
            # Convert float stake to int if it's a whole number (e.g., 1.0 -> 1)
            stake_display = int(stake) if stake.is_integer() else stake
            data_list.append({
                'stake': stake_display,
                'games': data['games'],
                'wins': data['wins']
            })

        # Convert to a pandas DataFrame and save as data.csv
        df = pd.DataFrame(data_list).sort_values(by='stake')
        df.to_csv('data.csv', index=False)

        print("\nSuccessfully generated 'data.csv':")
        print(df)
    else:
        print("\nNo valid 3-player game data was found. 'data.csv' not created.")