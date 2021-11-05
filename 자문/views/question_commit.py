from flask import Blueprint, url_for, request, render_template
from werkzeug.utils import redirect
from web import db
from web.models import Question, Answer

bp = Blueprint('question', __name__, url_prefix="/input")

@bp.route('/create', methods=('POST',))
def create() :
    content = request.form['content']
    question = Question(rank=0, content=content)
    db.session.add(question)
    db.session.commit()
    return redirect(url_for('main.result'))
