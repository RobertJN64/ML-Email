from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def createEmail(subject, text, sender, reciever, attachments):
    """Returns an email object"""

    #create normal email
    message_to_send = MIMEMultipart("alternative")
    message_to_send['Subject'] = subject
    message_to_send['From'] = sender
    message_to_send['To'] = reciever
    message_to_send.attach(MIMEText(text, "plain"))

    #add attachments
    for attachment in attachments:
        with open(attachment, 'rb') as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)

        filename = attachment[attachment.rfind("/")+1:]

        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message and convert message to string
        message_to_send.attach(part)

    return message_to_send.as_string()