# 화면 구성

from flask import Blueprint, render_template, request

bp = Blueprint('main', __name__, url_prefix='/')

# getQuestion.py 위치에 따라 수정해주세요
from . import getQuestion

@bp.route('/')
def hello() :
    return render_template('index.html')

@bp.route('/input')
def input() :
    return render_template('input.html')

@bp.route('/interview', methods=('GET', 'POST'))
def interview() :
    if request.method == 'POST' :
        es = request.form.get('content')
        q = getQuestion.get_question(es)
        return render_template('interview.html', question=q, question_n=len(q))

@bp.route('/introduce')
def introduce() :
    return render_template('introduce.html')
