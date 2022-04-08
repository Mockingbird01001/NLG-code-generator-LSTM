
from random import *
import json
import sys
max_of_p_q = 10000
def is_prime(num):
 if num == 0 or num == 1:
		return False
 for x in range(2, num//2):
		if num % x == 0:
			return False
 else:
		return True
def gcd(a, b):
 while b:
		a, b = b, a%b
 return a
def lcm(a, b):
	return (a*b) / gcd(a, b)
def is_coprime(a, b):
	if gcd(a, b) == 1:
		return True
 else:
		return False
def mod_mul_inv(a, num):
 for i in xrange(num):
		if (a*i)%num == 1:
			return i
 return 0
def generate_keys():
	p, q = 1, 1
 while(not is_prime(p)):
		p = randint(1, max_of_p_q)
 while(not is_prime(q) and p!=q):
		q = randint(1, max_of_p_q)
 n = p * q
 lambda_n = lcm(p-1, q-1)
 e = randint(1, lambda_n)
 while(not is_coprime(e, lambda_n)):
		e = randint(1, lambda_n)
 d = mod_mul_inv(e, lambda_n)
 public_key = (e, n)
 private_key = (d, n)
 return public_key, private_key
def encrypt(public_key, m):
	e = public_key[0]
 n = public_key[1]
 c = pow(m, e, n)
 return c
def decrypt(private_key, ciphertext):
	d = private_key[0]
 n = private_key[1]
 plaintext = pow(ciphertext, d, n)
 return plaintext
def main():
	case = int(raw_input("1 to generate keys and save in file\n2 to encrypt\n3 to decrypt\n"))
 if case == 1:
		public_key, private_key = generate_keys()
  print "Public Key (e, n) : ", public_key
  print "Private Key (d, n) : ", private_key
  pubkey = {}
  pubkey["e"] = public_key[0]
   pubkey["n"] = public_key[1]
  with open("public_key.json", "w") as pubkeyfile:
			json.dump(pubkey, pubkeyfile)
  prikey = {}
  prikey["d"] = private_key[0]
   prikey["n"] = private_key[1]
  with open("private_key.json", "w") as prikeyfile:
			json.dump(prikey, prikeyfile)
 elif case == 2:
		m = int(raw_input("Enter message num: "))
  with open("public_key.json", "r") as pubkeyfile:
			pubkey = json.load(pubkeyfile)
  public_key = (pubkey["e"], pubkey["n"])
  ciphertext = encrypt(public_key, m)
  print "Encrypted: ", ciphertext
 elif case == 3:
		ciphertext = int(raw_input("Enter Encrypted num: "))
  with open("private_key.json", "r") as prikeyfile:
			prikey = json.load(prikeyfile)
  private_key = (prikey["d"], prikey["n"])
  plaintext = decrypt(private_key, ciphertext)
  print "Decrypted: ", plaintext
 else:
		print "Invalid option\n"
  sys.exit(0)
if __name__ == "__main__":
	main()