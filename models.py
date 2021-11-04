# DB처리

from web import db

class Question(db.Model) :
    id = db.Column(db.Integer, primary_key=True)
    rank = db.Column(db.Integer, nullable = False)
    content = db.Column(db.Text(), nullable=False)

class Answer(db.Model) :
    id = db.Column(db.Integer, primary_key=True)
    question = db.relationship('Question')
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    mp3 = db.Column(db.Text(), nullable=True)
    content = db.Column(db.Text(), nullable=True)