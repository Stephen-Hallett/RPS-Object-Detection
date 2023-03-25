import os
import random
import json
from pprint import pprint


def decay(hist,decay=0.1):
    outcomes = ['w','l','d']
    for dict in hist:
        for outcome in outcomes:
            hist[dict][outcome] = hist[dict][outcome]*(1-decay)
    return hist

def normalize(hist):
    outcomes = ['w','l','d']
    for dict in hist:
        denominator = sum([hist[dict][outcome] for outcome in outcomes])
        for outcome in outcomes:
            hist[dict][outcome] = hist[dict][outcome]/denominator
    return hist

class RPS():
    def __init__(self,name):
        self.history = {}
        self.name = name
        self.user_moves = []
        self.computer_moves = []
        self.score = [0,0]
        self.user_winloss = []
        self.loses = {'r': 'p',
         'p': 's',
         's': 'r'}
        self.beats = {'r': 's',
            'p': 'r',
            's': 'p'}
        self.winloss = {'r': {'r':'d',
                 'p':'w',
                 's':'l'},
            'p': {'r':'l',
                'p':'d',
                's':'w'},
            's': {'r':'w',
                'p':'l',
                's':'d'}}
        self.outcomes = ['w','l','d']
        self.options = ['r','p','s']
        self.user_outcomes = ""
        self.get_history()

    def create_history(self):
        start_odds = 1/3
        history = {}
        for outcome in self.outcomes:
            history[outcome] = {self.outcomes[i]:start_odds for i in range(len(self.outcomes))}
            for outcome2 in self.outcomes:
                history[outcome+outcome2] = {self.outcomes[i]:start_odds for i in range(len(self.outcomes))}
        return history

    def get_history(self):
        if os.path.isfile(os.path.join("history",self.name+"_history_winloss.json")):
            file = os.path.join("history",self.name+"_history_winloss.json")
            with open(file) as jsonfile:
                self.history = json.load(jsonfile)
                jsonfile.close()
        else:
            if not os.path.isdir("history"):
                os.mkdir("history")
            self.history = self.create_history()

    def save_history(self):
        jsondict = json.dumps(self.history)
        file = os.path.join("history", self.name+"_history_winloss.json")
        with open(file, 'w') as f:
            print('writing history to',self.name+"_history_winloss.json")
            f.write(jsondict)
            f.close()

    def update_winloss(self):
        self.user_winloss = [None for i in self.user_moves]
        for i in range(1,len(self.user_moves)):
            winloss = self.winloss[self.user_moves[i-1]][self.user_moves[i]]
            self.user_winloss[i] = winloss

    def update_history(self, lr=0.02):
        #for every move the learner makes, see if it loses, beats to or draws with the move they made before
        self.update_winloss()

        if len(self.user_winloss) > 1: #need it to be of form [None, x]
            outcomes = self.user_outcomes
            history = decay(self.history)
            history[outcomes[-2]][self.user_winloss[-1]] += lr
            if len(self.user_winloss) > 2:
                history[outcomes[-3]+outcomes[-2]][self.user_winloss[-1]] += lr
            self.history = normalize(history)

    def predict(self):
        game_state = self.user_outcomes[-2:]
        if len(game_state) == 0:
            user_pick = ['r','p','s'][random.randrange(3)]
        else:
            lastplayed = self.user_moves[-1]
            prediction = max(self.history[game_state], key=self.history[game_state].get)
            if prediction == 'd':
                user_pick = lastplayed
            elif prediction == 'l':
                user_pick = self.loses[lastplayed]
            else:
                user_pick = self.beats[lastplayed]
        return self.loses[user_pick]

    def save_data(self):
        filename = self.name + "_computer"
        count = len([file for file in os.listdir("rps_data") if filename in file])
        filename = os.path.join("rps_data",filename+"_{}.txt".format(count))
        print('Writing data to {}'.format(filename))
        data = [self.user_moves[i]+self.computer_moves[i] for i in range(len(self.user_moves))]
        with open(filename, 'w') as f:
            for line in data:
                f.write(line+'\n')
            f.close()
    
    def update_score(self):
        char = self.user_outcomes[-1]
        if char == "w":
            self.score[0] += 1
            print("You win!")
        elif char == "l":
            self.score[1] += 1
            print("You lose!")
        else:
            print("Draw!")

    def played(self,user_played,prediction):
        if user_played in self.options:
            self.user_moves.append(user_played)
            self.computer_moves.append(prediction)
            self.user_outcomes += self.winloss[prediction][user_played]
            self.update_history()
            self.update_score()
            #pprint(self.history)
            return self.winloss[prediction][user_played]
        else:
            print("ERROR: \"{}\" IS NOT A VALID INPUT".format(user_played))
            return "e"

    def quit(self):
        self.save_data()
        self.save_history()

    def __str__(self):
        return "{} vs Computer: {}-{}".format(self.name,self.score[0],self.score[1])
