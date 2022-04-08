
import sys
alphabets = list('abcdefghijklmnopqrstuvwxyz.,!')
len_alpha = len(alphabets)
def remove_punctuation(message):
	message = message.translate(None, punctuation);
 message = ''.join(message.split())
 return message.lower()
def padthemessage(message):
	if(len(message)%3 == 0):
		return message
 else:
		short = 3 - len(message)%3
  for i in range(short):
			message += 'x'
  return message
def display_format(message):
 output = []
 enc = ""
 while message:
		output.append(message[:5])
  message = message[5:]
 for index, block in enumerate(output):
		if((index + 1) % 10 == 0):
			enc = enc + block + "\n"
  else:
			enc = enc + block + " "
 return enc
def modular_multiplicative_inverse(number):
	for i in range(len_alpha):
		if (number*i)%len_alpha == 1:
			return i
 return 0
def matrix_transpose(matrix):
	return [list(x) for x in zip(*matrix)]
def matrix_minor(matrix,i,j):
	return [row[:j] + row[j+1:] for row in (matrix[:i]+matrix[i+1:])]
def matrix_determinant(matrix):
	if len(matrix) == 2:
		return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
 determinant = 0
 for c in range(len(matrix)):
		determinant += ((-1)**c)*matrix[0][c]*matrix_determinant(matrix_minor(matrix,0,c))
 return determinant
def key_matrix_inverse(matrix):
	determinant = matrix_determinant(matrix)
 cofactors = []
 for r in range(len(matrix)):
		cofactorRow = []
  for c in range(len(matrix)):
			minor = matrix_minor(matrix,r,c)
   cofactorRow.append(((-1)**(r+c)) * matrix_determinant(minor))
  cofactors.append(cofactorRow)
 cofactors = matrix_transpose(cofactors)
 determinant = modular_multiplicative_inverse(determinant)
 for r in range(len(cofactors)):
		for c in range(len(cofactors)):
			cofactors[r][c] = (cofactors[r][c]*determinant)%len_alpha
 return cofactors
def matrix_multiply(matrix_1, matrix_2):
	if(len(matrix_1[0]) == len(matrix_2)):
		rows = len(matrix_1)
  cols = len(matrix_2[0])
  cipher_matrix = [[0]]*rows
  for i in range(rows):
			row = []
   for j in range(cols):
				res = 0
    for k in range(len(matrix_2)):
					res += (matrix_1[i][k] * matrix_2[k][j])
    row.append(alphabets[res%len_alpha])
   cipher_matrix[i] = row
  cipher_matrix = matrix_transpose(cipher_matrix)
  return cipher_matrix
 else:
		print "Matrices cannnot be multiplied..."
  sys.exit(0);
def generate_matrix(content, rows, cols):
	matrix = [[0]]*rows
 k = 0
 for i in range(rows):
		row = []
  for j in range(cols):
			row.append(alphabets.index(content[k]))
   k+=1
  matrix[i] = row
 return matrix
def encrypt(message, key):
	key_matrix = generate_matrix(key, 3, 3)
 if(matrix_determinant(key_matrix) == 0):
		print "the determinant of your key is 0. you won't be able to decrypt. use another key."
  sys.exit(0)
 message_matrix = generate_matrix(message, len(message)/3 , 3)
 message_matrix = matrix_transpose(message_matrix)
 cipher_matrix = matrix_multiply(key_matrix, message_matrix)
 ciphertext = ''.join([ch for row in cipher_matrix for ch in row])
 ciphertext = display_format(ciphertext)
 return ciphertext
def decrypt(ciphertext, key):
	key_matrix = generate_matrix(key, 3, 3)
 key_matrix = key_matrix_inverse(key_matrix)
 cipher_matrix = generate_matrix(ciphertext, len(ciphertext)/3 , 3)
 cipher_matrix = matrix_transpose(cipher_matrix)
 message_matrix = matrix_multiply(key_matrix, cipher_matrix)
 plaintext = ''.join([ch for row in message_matrix for ch in row])
 return plaintext
def main():
	try:
		case = int(raw_input("1 to encrypt plain_text\n2 to decrypt ciphertext\nEnter option : "))
  if(case == 1):
			k = raw_input("Enter key (exactly 9 chars): ")
   if(len(k) != 9):
				print("Enter key of size 9 only...")
    sys.exit(0)
   message = padthemessage(remove_punctuation(raw_input("Message[Plain Text]: ")))
   encrypted_message = encrypt(message, k)
   print "Encrypted Message..."
   print encrypted_message
   sys.exit(0)
  if(case == 2):
			k = raw_input("Enter key (exactly 9 chars): ")
   if(len(k) != 9):
				print("Enter key of size 9 only...")
    sys.exit(0)
   encrypted_message = remove_punctuation(raw_input("Ciphertext: "))
   plain_text = decrypt(encrypted_message, k)
   print "Decrypted Message..."
   print plain_text
   sys.exit(0)
  else:
			print "Wrong option"
   sys.exit(0)
 except KeyboardInterrupt:
		print "\nClosing the program..."
  sys.exit(0)
if __name__ == '__main__':
	main()
