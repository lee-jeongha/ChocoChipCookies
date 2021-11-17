# 화면 구성

from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')
q = ['자기소개를 해보세요. ', '자원봉사를 하면서 소모임장을 했다고 했는데, 모임장을 하게 된 동기는?', '팀 과제나 프로젝트를 하면서 어려움이 있었던 경우에 대해 말해보세요.']

@bp.route('/')
def hello() :
    return render_template('index.html')

@bp.route('/input')
def input() :
    return render_template('input.html')

@bp.route('/interview')
def interview() :
    return render_template('interview.html', question=q, question_n=len(q))
