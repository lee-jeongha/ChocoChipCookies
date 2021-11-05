# 화면 구성

from flask import Blueprint, render_template
from web.models import Question

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def hello() :
    return render_template('index.html')

@bp.route('/test')
def test() :
    return render_template('test.html')

@bp.route('/input')
def input() :
    return render_template('input.html')

@bp.route('/result')
def result() :
    question_list = Question.query.order_by("id").all()
    return render_template('result.html', question_list = question_list)

@bp.route('/record')
def record() :
    return render_template('record.html')
