from flask import Flask,render_template,request
from flask_cors import cross_origin
import pandas as pd
import pickle

app=Flask(__name__)

@app.route('/',methods=["GET"])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/Prediction',methods=["GET","POST"])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            potential=float(request.form['potential'])
            foot=int(request.form['foot'])
            attacking_work_rate=request.form['attacking_work_rate']
            defensive_work_rate=request.form['defensive_work_rate']
            crossing=float(request.form['crossing'])
            finishing=float(request.form['finishing'])
            heading_accuracy=float(request.form['heading_accuracy'])
            volleys=float(request.form['volleys'])
            curve=float(request.form['curve'])
            free_kick_accuracy=float(request.form['free_kick_accuracy'])
            long_passing=float(request.form['long_passing'])
            ball_control=float(request.form['ball_control'])
            sprint_speed=float(request.form['sprint_speed'])
            agility =float(request.form['agility'])
            reactions=float(request.form['reactions'])
            balance=float(request.form['balance'])
            shot_power=float(request.form['shot_power'])
            jumping=float(request.form['jumping'])
            stamina=float(request.form['stamina'])
            strength=float(request.form['strength'])
            long_shots=float(request.form['long_shots'])
            aggresion=float(request.form['aggression'])
            interceptions=float(request.form['interceptions'])
            positioning=float(request.form['positioning'])
            vision=float(request.form['vision'])
            penalties=float(request.form['penalties'])
            sliding_tackle=float(request.form['sliding_tackle'])
            gk_kicking=float(request.form['gk_kicking'])
            gk_reflexes=float(request.form['gk_reflexes'])

            data=[[potential,foot,attacking_work_rate,defensive_work_rate,crossing,finishing,heading_accuracy,
                   volleys,curve,free_kick_accuracy,long_passing,ball_control,sprint_speed,
                   agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots
                   ,aggresion,interceptions,positioning,vision,penalties,sliding_tackle,gk_kicking,gk_reflexes]]

            data=pd.DataFrame(data,columns=['potential', 'preferred_foot', 'attacking_work_rate',
                               'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy',
                               'volleys', 'curve', 'free_kick_accuracy', 'long_passing',
                               'ball_control', 'sprint_speed', 'agility', 'reactions', 'balance',
                               'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
                               'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
                               'sliding_tackle', 'gk_kicking', 'gk_reflexes'])

            attacking_rate_map={'medium': 126723,'high': 43403,'low': 8674,'norm': 353}
            data.attacking_work_rate=data.attacking_work_rate.map(attacking_rate_map)

            defensive_rate_map = {'medium': 130846,'high': 27041, 'low': 18432,'ormal': 348}
            data.defensive_work_rate = data.defensive_work_rate.map(defensive_rate_map)

            model=pickle.load(open('xgboost.pickle' , 'rb'))
            pred=model.predict(data)

            return render_template("results.html",prediction=pred[0])
        except Exception as e:
            return print(e)
    else:
        return render_template("results.html")



if __name__=='__main__':
    app.run(debug=True,host='127.0.0.1',port=8001)