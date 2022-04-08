from flask import url_for, render_template
import smtplib
import ssl
import configparser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from verification import generate_confirmation_token
def send_email(receiver_email, subject, plaintext, html):
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    email_config = config['EMAIL']
    SMTP_SERVER = email_config['SMTP_SERVER']
    PORT = 587
    SENDER_EMAIL = email_config['SENDER_EMAIL']
    PASSWORD = email_config['PASSWORD']
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email
    part1 = MIMEText(plaintext, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(SMTP_SERVER, PORT)
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(SENDER_EMAIL, PASSWORD)
        server.send_message(message)
    except Exception as e:
        print(e)
        return False
    finally:
        server.quit()
        return True
def send_registration_email(user):
    token = generate_confirmation_token(user.email)
    confirm_url = url_for('confirm_email', token=token, _external=True)
    subject = "Registration successful - Please verify your email address."
    plaintext = f"Welcome {user.display_name()}.\nPlease verify your email address by following this link:\n\n{confirm_url}"
    html = render_template('verification_email.html',
                           confirm_url=confirm_url, user=user)
    send_email(user.email, subject, plaintext, html)
def send_message_email(from_user, to_user, message):
    subject = f"{from_user.display_name()} sent you a message"
    plaintext = f"{to_user.display_name()} sent you this message:\n\n{message.title}\n\n{message.body}"
    html = render_template(
        'message_email.html', from_user=from_user, to_user=to_user, message=message)
    send_email(to_user.email, subject, plaintext, html)
