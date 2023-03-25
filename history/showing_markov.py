import json
from pprint import pprint

file = "history/stephen_history_winloss.json"
with open(file) as jsonfile:
    history = json.load(jsonfile)
    jsonfile.close()

pprint(history)