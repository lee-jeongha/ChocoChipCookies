from flask import Flask, render_template, request
import summarize

# flask 앱 서버 인스턴스
app = Flask(__name__)

# url 패턴 - 라우터 설정 - 데코레이터
@app.route('/')
def index():
    # get을 통해 전달받은 데이터 확인
    text = request.args.get('text')

    if not text :
        return render_template('index.html')
    else :
        # 모듈로 요약문 가져오기
        summm = summarize.sum(text)

        # 사용자에게 보낼 데이터
        data = summm

        return render_template('index.html', data=data, value = text)

# 메인 테스트
if __name__ == "__main__" :
    app.run('0.0.0.0', port=5000, debug=True)
