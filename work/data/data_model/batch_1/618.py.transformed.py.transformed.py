
from flask import Flask, request
from flask_mongoalchemy import MongoAlchemy
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['MONGOALCHEMY_DATABASE'] = 'library'
db = MongoAlchemy(app)
class Author(db.Document):
    name = db.StringField()
class Book(db.Document):
    title = db.StringField()
    author = db.DocumentField(Author)
    year = db.IntField()
@app.route('/author/new')
def new_author():
    author = Author(name=request.args.get('name', ''))
    author.save()
    return 'Saved :)'
@app.route('/authors/')
def list_authors():
    authors = Author.query.all()
    content = '<p>Authors:</p>'
    for author in authors:
        content += '<p>%s</p>' % author.name
    return content
if __name__ == '__main__':
    app.run()
