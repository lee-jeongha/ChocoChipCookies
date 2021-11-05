from flask import Flask, render_template, redirect, request, url_for
import requests
import json
import os
import wave
import synthesize


app = Flask(__name__, static_folder='static')
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods = ["GET"])
def ttsTest():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def text_to_speech():
    text = request.form['text']
    voice = request.form['voice']

    key = '6a8aaf87489d0e1174e3217e0b84111a'
    url = "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize"
    headers = {
        "Content-Type": "application/xml",
        "Authorization": "KakaoAK " + key,
    }
    data = "<speak><voice name=\"" + voice + "\">" + text.encode('utf-8').decode(
        'latin1') + "</voice></speak>"

    res = requests.post(url, headers=headers, data=data)
    url_mp3 = "C:\\flask\\tts\\static\\" + text + ".mp3"
    f=open(url_mp3, "wb")
    f.write(res.content)
    f.close()
    text = text + ".mp3"
    return render_template('results.html', text=text)

if __name__ == '__main__':
    app.run()
