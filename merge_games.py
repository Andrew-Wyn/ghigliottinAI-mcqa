import json

GAME_1_PATH=""
GAME_2_PATH=""
OUT_PATH=""

with open(GAME_1_PATH, "r") as f:
    game_1 = json.load(f)

with open(GAME_2_PATH, "r") as f:
    game_2 = json.load(f)

print(f"Check number of repeated games-> ({len([x for x in game_1 if x in game_2])})")

game_merged = game_1 + game_2

with open(OUT_PATH, "w") as f:
    f.writelines(json.dumps(game_merged, indent=4))
