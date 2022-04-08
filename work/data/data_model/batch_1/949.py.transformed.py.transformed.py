
import bcrypt
passw = 'secret'
my_str_as_bytes = str.encode(passw)
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(my_str_as_bytes, salt)
print(str.decode(hashed))
if bcrypt.checkpw(my_str_as_bytes, hashed):
    print("match")
else:
    print("does not match")