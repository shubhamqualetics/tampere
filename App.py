#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Model11.pkl', 'rb'))

d = {'06-03-2020':10,'07-03-2020':25,'08-03-2020':35,'09-03-2020':40,'10-03-2020':59,'11-03-2020':60,'12-03-2020':78,'13-03-2020':80,
    '14-03-2020':10,'15-03-2020':20,'16-03-2020':30,'17-03-2020':40,'18-03-2020':80,'19-03-2020':100,'20-03-2020':50,'21-03-2020':30,
 '22-03-2020':30,'23-03-2020':70,'24-03-2020':80,'25-03-2020':18,'26-03-2020':90,'27-03-2020':92,'28-03-2020':60,'29-03-2020':95,
'30-03-2020':40,'31-03-2020':27,'01-04-2020':39,'02-04-2020':40,'03-04-2020':80,'04-03-2020':34,'05-03-2020':88,'06-03-2020':88,
'07-04-2020':50,'08-04-2020':21,'09-04-2020':49,'10-04-2020':70,'11-04-2020':90,'12-04-2020':79,'13-04-2020':66,'15-04-2020':39} 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    aa=int(int_features[0])
    bb=int(int_features[1])
    cc=int(int_features[2])
    dd=int_features[3]
    change = ""
    if dd in d:
    	change = d[dd]
    else:
    	change=10
    change=int(change/10)
    apple=[aa,bb,cc,change]
    final_features = [np.array(apple)]
    prediction = model.predict(final_features)
    output = round(prediction[0])

    return render_template('index.html', prediction_text= 'Total delay is {} seconds approx'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




