"""
Author: Lin Shaoqing
MIT License
"""

import smtplib
from email.mime.text import MIMEText
from email.header import Header

"""
use your mailbox to send a mail to notify you
from_addr = 'your_mail_account'
password = 'your_smtp_password'
eg: mailme = Mailme('your_mail_account', 'your_smtp_password, 'receive_mail_account')
    mailme.send_me_mail()
"""

class Mailme:
    def __init__(self, from_addr, password, to_addr, mailbox_name):
        """
        Argument:
            from_addr: mailbox sending mails
            password: your smtp password
            to_addr: mailbox receiving mails
            mailbox_name: now only support(gmail, qqmail, 163mail, outlook)
                          please use '163', 'gmail', 'qq' or 'outlook' as your input
        """
        self.from_addr = from_addr
        self.password = password
        self.to_addr = to_addr
        self.mailbox_name = mailbox_name

    def send_me_text_mail(self, head='Train Finished!', text='Go to check your result!'):
        """
        send an email from 'from_addr' to 'to_addr' using the password 'password'
        """
        mailboxs_dict = {
            '163': ['smtp.163.com', 465],
            'gmail': ['smtp.gmail.com', 587],
            'qq': ['smtp.qq.com', 25],
            'outlook': ['outlook.office365.com', 587],
        }
        smtp_server = mailboxs_dict[self.mailbox_name][0]
        msg = MIMEText(text, 'plain', 'utf-8')
        """s
        input what you want to send above, only text now ;]
        """

        msg['From'] = Header(self.from_addr)
        msg['To'] = Header(str(self.to_addr))
        msg['Subject'] = Header(head)
        """
        input mail subject above
        """

        server = smtplib.SMTP_SSL(smtp_server)
        server.connect(smtp_server, mailboxs_dict[self.mailbox_name][1])
        server.login(self.from_addr, self.password)
        server.sendmail(self.from_addr, self.to_addr, msg.as_string())
        server.quit()
