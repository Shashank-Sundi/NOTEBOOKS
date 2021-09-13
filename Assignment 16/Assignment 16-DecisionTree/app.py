from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle

url = "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic=pd.read_csv(url)
sex_map={'male':1,'female':0}
Pclass_map=titanic.Pclass.value_counts().to_dict()

app = Flask(__name__)

@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template('index.html')

@app.route('/Prediction',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            Pclass=int(request.form['Pclass'])
            Sex=request.form['sex']
            Age=int(request.form['Age'])
            SibSp=int(request.form['SibSP'])
            Parch=int(request.form['Parch'])
            Fare=float(request.form['Fare'])

            a = [[Pclass,Sex,Age,SibSp,Parch,Fare]]
            data = pd.DataFrame(data=a,columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
            data.Pclass=data.Pclass.map(Pclass_map)
            data.Sex=data.Sex.map(sex_map)

            dtree=pickle.load(open('DecisionTreeClassifier.pickle','rb'))
            prediction=dtree.predict(data)

            if prediction==1:
                decision='survived the accident'
            else :
                decision="didn't survive the accident"

            return render_template('results.html', prediction=decision)
        except Exception as e:
            return print(e)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8001)
