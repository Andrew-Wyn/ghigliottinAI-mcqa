import xmltodict
import json

XML_PATH=""
OUT_PATH=""


with open(XML_PATH, "r") as f:
    dict_games = xmltodict.parse("\n".join(f.readlines()))

out_obj_games = []
for game in dict_games["games"]["game"]:
    #if game["type"] == "TV": # only tv games
    out_obj_games.append(
        {
            "w1": game["clue"][0],
            "w2": game["clue"][1],
            "w3": game["clue"][2],
            "w4": game["clue"][3],
            "w5": game["clue"][4],
            "solution": game["solution"]
        }
    )


with open(OUT_PATH, "w") as f:
    f.writelines(json.dumps(out_obj_games, indent=4))