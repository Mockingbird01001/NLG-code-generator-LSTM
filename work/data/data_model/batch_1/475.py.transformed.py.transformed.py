import ftplib
def download_file(filename):
    localfile = open(filename, 'wb')
    ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
    ftp.quit()
    localfile.close()
def upload_file(filename):
    ftp.storbinary('STOR ' + filename, open(filename, 'rb'))
    ftp.quit()
ftp = ftplib.FTP(host='10.10.1.111')
print('Connect')
ftp.login(user='weit', passwd='weit2.71')
print('Login')
ftp.cwd('/home/')
upload_file('test.txt'), print('Upload')
download_file('test.txt'), print('Download')
