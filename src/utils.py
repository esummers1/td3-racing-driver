import locale

from datetime import datetime


def render_int(num):
    _init_locale()
    return locale.format("%d", round(num), grouping=True)


def render_decimal(num):
    _init_locale()
    return locale.format("%d", round(num, 2), grouping=True)


def render_time(seconds):
    minutes = seconds / 60
    hours = minutes / 60

    if hours < 1:
        if minutes < 1:
            return f"{render_int(seconds)} seconds"
        else:
            return f"{render_int(minutes)} minutes"
    else:
        return f"{render_decimal(hours)} hours"


def _init_locale():
    locale.setlocale(locale.LC_ALL, 'en_GB.utf8')


def datetime_now():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")
