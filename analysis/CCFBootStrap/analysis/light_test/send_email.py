from datetime import datetime, timezone, timedelta
import platform
import smtplib
import getpass
import email


def send_email(send=True):
    def _send_email(func):
        def wrapper(*args, **kargs):

            # set flag
            send_flag = send

            # get address and password
            if send_flag:
                from_address = 'tomama0920@gmail.com'
                to_address = 'tomama0920@gmail.com'
                # from_address = input('From (gmail): ')
                # to_address = input('To (gmail): ')
                password = getpass.getpass(prompt='Passward for gmail:')

            # email won't send if  no password
            if password == '':
                print("Mail won't send because of no password")
                send_flag = False

            # send email
            if send_flag:
                # check SMTP server
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(from_address, password)
                print('SMTP server: CLEAR')

                try:
                    # start analysis
                    JST = timezone(timedelta(hours=+9), 'JST')
                    start_date = datetime.now(JST)
                    result = func(*args, **kargs)
                    end_time = datetime.now(JST)

                    # compose content
                    content = f'Function: {func.__name__}\n'
                    content += time.strftime('%Y-%m-%d %H:%M:%S START ANALYSIS\n',
                                             time.gmtime(start_date))
                    content += time.strftime('%Y-%m-%d %H:%M:%S FINISH ANALYSIS\n',
                                             time.gmtime(end_date))
                    content += f'\nConducted by {platform.system()}'

                # send error log if it occured
                except Exception as error:
                    content = f'Function: {func.__name__}\n'
                    content += 'Error occured\n'
                    content += '\n===== ERROR CONTENT IS BELOW =====\n'
                    content += str(error)

                # make emain
                msg = email.message.EmailMessage()
                msg.set_content(content)
                msg['Subject'] = f'[PYTHON] Result of {func.__name__}'
                msg['From'] = from_address
                msg['To'] = to_address

                # send message
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(from_address, password)
                server.send_message(msg)
                server.quit()

            # flag is not 1, do analysis without sending mail
            elif send_flag is False:
                result = func(*args, **kargs)

            return result
        return wrapper
    return _send_email
