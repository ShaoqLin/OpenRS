from mailMe import Mailme
import time

while True:
    try:
        mailme = Mailme('kyrie_lin@163.com', 'TENOTYLLYMLXDPBH', 'kyrie_lin@163.com', '163')
        mailme.send_me_text_mail(head='all done!', text='Go check your result')
        print('Mailme: Success!')
        break
    except:
        print('Mailme: Failed, retry in 60s...')
        time.sleep(60)
        continue
