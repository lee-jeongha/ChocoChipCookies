from flask import Flask, render_template, request

app = Flask(__name__, static_folder='static')



@app.route('/', methods = ["GET"])
def sendText():
    return render_template('getResume.html')



@app.route('/', methods = ["POST"])
def nextQues():
    resume = ""

    if request.method == 'POST':
        resume = request.form.get("resume", "홍길동")

    resume_list = resume.split('.')

    return render_template('nextQuestion.html', resume=resume, resume_list=resume_list)



if __name__ == "__main__":
    app.run(debug=True)