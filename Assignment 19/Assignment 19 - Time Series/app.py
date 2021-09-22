from flask import Flask,request,render_template
from flask_cors import cross_origin
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import datetime

def parser(x):
    return datetime.datetime.strptime('19'+x, '%Y-%m')

series =pd.read_csv('https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/sales-of-shampoo-over-a-three-ye.csv',sep=';', header=0, parse_dates=[0],index_col=0, squeeze=True,date_parser=parser)
series=pd.DataFrame(series,columns=['Sales'])

app=Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/Prediction',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            start_year=int(request.form['start_year'])
            end_year=int(request.form['end_year'])
            start_month=int(request.form['start_month'])
            end_month=int(request.form['end_month'])

            b=(datetime.datetime(end_year,end_month,1)+datetime.timedelta(days=31)).strftime("%Y-%m-%d")
            a = datetime.datetime(start_year, start_month, 1).strftime("%Y-%m-%d")
            FMT = '%Y-%m-%d'
            time_diff = datetime.datetime.strptime(b, FMT) - datetime.datetime.strptime(a, FMT)
            time_diff = np.round(time_diff.days / 30)

            end_date = b
            start_date = datetime.datetime(1904, 1, 1).strftime("%Y-%m-%d")
            FMT = '%Y-%m-%d'
            tdelta = datetime.datetime.strptime(end_date, FMT) - datetime.datetime.strptime(start_date, FMT)

            history = [x for x in series.Sales]
            predictions = []
            for t in range(int(np.round(tdelta.days / 30))):
                if t==int(np.round(tdelta.days / 30))-1:
                    break
                model = ARIMA(history, order=(4, 2, 1))
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                predictions.append(yhat[0])
                history.append(yhat)

            d = pd.DataFrame(data={"Predictions": predictions}, index=pd.date_range(start_date, end_date, freq='M'))
            d.index = d.index.to_period('M')

            pred=d.iloc[-int(time_diff):]

            return render_template("results.html",tables=[pred.to_html(classes='data')], titles=pred.columns.values)
        except Exception as e :
            return print(e)
    else :
        return render_template("index.html")

if __name__=='__main__':
    app.run(debug=True,host='127.0.0.1',port=8001)