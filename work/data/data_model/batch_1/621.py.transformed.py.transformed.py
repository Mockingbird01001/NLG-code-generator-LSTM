from flask.view import MethodView
from flask.ext.mongoalchemy import MongoAlchemy
from flask.ext.login import login_required, current_user
class LikeAPI(MethodView):
	def __init__(self, collection):
		self.collection = collection
 def put(self,user_id):
		try:
			obj = self.collection.query.get_or_404(user_id)
   obj.likes.append(current_user)
   obj.save()
  except MongoAlchemy.exceptions.MissingValueException as e:
			return 'The document has no like field'
