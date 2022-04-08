
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath('_themes'))
extensions = ['sphinx.ext.autodoc']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'Flask MongoAlchemy'
copyright = u'2015, Francisco Souza'
version = '0.7'
release = '0.7.2'
exclude_patterns = ['_build']
html_theme = 'flask_small'
html_theme_options = {
    'index_logo': 'flask-mongoalchemy.png',
    'github_fork': 'cobrateam/flask-mongoalchemy'
}
html_theme_path = ['_themes']
html_static_path = ['_static']
htmlhelp_basename = 'FlaskMongoAlchemydoc'
latex_documents = [
    ('index', 'FlaskMongoAlchemy.tex', u'Flask MongoAlchemy Documentation',
     u'Francisco Souza', 'manual'),
]
man_pages = [
    ('index', 'flaskmongoalchemy', u'Flask MongoAlchemy Documentation',
     [u'Francisco Souza'], 1)
]
