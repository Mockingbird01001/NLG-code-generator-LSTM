from flask.views import View
from flask_login import current_user, login_required
from flask_mongoalchemy import MongoAlchemy
from flask import request, redirect
class LikeAPI(View):
	decorators=[login_required]
 def __init__(self, collection, user_col, *args, **kwargs):
		super(LikeAPI, self).__init__(*args,**kwargs)
  self.collection = collection
  self.user_col = user_col
 def dispatch_request(self,id):
		try:
			obj = self.collection.query.get_or_404(id)
   usr = self.user_col.query.filter({'username':current_user.username}).first()
   if usr not in obj.likes_users:
				obj.likes_users.append(usr)
    obj.save()
   return redirect("/".join(request.full_path.split('/')[:-2]))
  except AttributeError as e:
			return '{0} does not exists.'.format(e)
