
import os
import jinja2
def render(filename, context):
    path = os.path.dirname(os.path.abspath(__file__))
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)
