""" Notifications via, e.g., email """


def send_simple_mail(message: str, to_addr: str, extra_message: str = None) -> None:
    """Send simple e-mail message

    Args:
      message: the message
      to_addr: the mail address of recipient
      extra_message:  prefix of message

    """
    import os

    log = os.system(
        'echo "{}" | mailx -s "[vibes] {:s}" {:s}'.format(
            extra_message, message, to_addr
        )
    )

    if log:
        print("Sending the Mail returned error code {:s}".format(str(log)))
