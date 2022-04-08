from myapp import app
from flask_login import current_user
from accounts.models import User
from utils import is_json
import urllib, urllib2, json
