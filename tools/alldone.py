from mailMe import Mailme

mailme = Mailme('kyrie_lin@163.com', 'TENOTYLLYMLXDPBH', 'kyrie_lin@163.com', '163')
mailme.send_me_text_mail(head='all done!',
                         text='Go check your result')
