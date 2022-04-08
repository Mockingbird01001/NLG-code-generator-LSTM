
from mongoengine.document import Document
from mongoengine.fields import DateTimeField, IntField, StringField, URLField
class Post(Document):
    url = URLField(required=True)
    date = DateTimeField(required=True)
    date_str = StringField(max_length=10, required=True)
    commentsUrl = URLField(required=True)
    sub = StringField(max_length=20, required=True)
    title = StringField(max_length=300, required=True)
    score = IntField(required=True)
    meta = {
        'collection': 'top_reddit_posts',
        'ordering': ['-score'],
        'auto_create_index': False,
        }
