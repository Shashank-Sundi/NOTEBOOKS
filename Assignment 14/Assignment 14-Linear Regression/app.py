

from flask import Flask, render_template , request
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from sklearn.datasets import load_boston

boston = load_boston()
bos = pd.DataFrame(boston.data)
columns=boston.feature_names
bos.columns=columns

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method=='POST' :
        try:
            CRIM=float(request.form['CRIM'])
            ZN=float(request.form['ZN'])
            INDUS=float(request.form['INDUS'])
            if request.form['CHAS']=='yes':
                CHAS=1
            else : CHAS=0
            NOX=float(request.form['NOX'])
            RM=float(request.form['RM'])
            AGE=request.form['AGE']
            DIS=float(request.form['DIS'])
            RAD=request.form['RAD']
            TAX=float(request.form['TAX'])
            PTRATIO=float(request.form['PTRATIO'])
            B=float(request.form['B'])
            LSTAT=float(request.form['LSTAT'])

            data=[[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]]
            scaler=StandardScaler()
            #data= np.power(data,0.5)
            scaler.fit(bos)
            data= scaler.transform(data)

            LinearRegModel=pickle.load(open('LinearReg.pickle','rb'))
            prediction=LinearRegModel.predict(data)
            print(f"The estimated price of the House is : {round(prediction[0]*1000)} dollars")
            return render_template('results.html',prediction = round(1000*prediction[0]))
        except Exception as e:
            print(f'Error Message : {e}')
            return  print(f'Error Message : {e}')
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(host='127.0.0.1', port=8001,debug=True)






