from flask import Flask

'''
def create_app() :
    app = Flask(__name__, static_folder='./static')

    # BluePrint
    import main_views
    app.register_blueprint(main_views.bp)

    return app
'''
app = Flask(__name__, static_folder='./static')

# BluePrint
From . import main_views
app.register_blueprint(main_views.bp)

