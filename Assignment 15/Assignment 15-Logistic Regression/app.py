from flask import Flask , render_template , request
from flask_cors import cross_origin
import pickle
import pandas as pd


app=Flask(__name__)

"""affair=sm.datasets.fair.load_pandas().data
affair['affair'] = (affair.affairs >0).astype(int)
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ +C(occupation) + C(occupation_husb)',affair, return_type="dataframe")
X = X.rename(columns ={'C(occupation)[T.2.0]':'occ_2','C(occupation)[T.3.0]':'occ_3','C(occupation)[T.4.0]':'occ_4','C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

scaler=StandardScaler()
scaler.fit(X)"""

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/Prediction',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method=='POST' :
        try:
            rate_marriage = int(request.form['Rate_Marriage'])
            Age=int(request.form['Age'])
            YearsMarried=int(request.form['YearsMarried'])
            Children=int(request.form['Children'])
            Religious=int(request.form['Religious'])
            Education=int(request.form['Education'])
            WomansOccupation=int(request.form['WomansOccupation'])
            HusbandsOccupation=int(request.form['HusbandsOccupation'])

            a = [[1,rate_marriage, Age, YearsMarried , Children, Religious,Education, WomansOccupation, HusbandsOccupation]]
            data=pd.DataFrame(data=a,columns=['Intercept','rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ','occupation', 'occupation_husb'])

            for i in range(4) :
                data.loc[len(data.index)]=a[0]

            for c in ['occupation', 'occupation_husb']:
                    occs = [2, 3, 4, 5, 6]
                    for i in range(data.shape[0]):
                        if occs.count(data[c][i])==1:
                            occs.remove(data[c].iloc[i])
                        else :
                            data[c][i]=occs[0]
                            occs.remove(occs[0])

            data=pd.get_dummies(data=data,columns=['occupation','occupation_husb'])
            data=data[['Intercept',  'occupation_2', 'occupation_3', 'occupation_4','occupation_5', 'occupation_6', 'occupation_husb_2','occupation_husb_3', 'occupation_husb_4', 'occupation_husb_5','occupation_husb_6','rate_marriage', 'age', 'yrs_married', 'children','religious', 'educ',]]

            data=pd.DataFrame(data.iloc[0]).T
            scaler=pickle.load(open('StandardScaler.pickle','rb'))
            LogisticRegression=pickle.load(open('LogisticRegression.pickle','rb'))
            data=scaler.transform(data)
            prediction=LogisticRegression.predict(data)
            print(prediction[0])

            if prediction[0]==1:
                decision='having'
            else:
                decision='not having'

            return render_template('results.html',prediction=decision)
        except Exception as e:
            return print(e)
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True , host='127.0.0.1',port=8001)
