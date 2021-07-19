import logging
from flask import Flask,request,redirect
import base64



app=Flask(__name__)



@app.route('/', methods=['GET','POST'])
def _imageClassfication():
    data=request.json
    imageData=data['imageData']
    try:
       decoed_data= base64.b64decode(imageData)
       #app.logger.info('----data decoded sucessfully-----')
       logging.info('----data decoded sucessfully-----')
       return decoed_data
    except :
        app.logger.error()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81,debug=True)