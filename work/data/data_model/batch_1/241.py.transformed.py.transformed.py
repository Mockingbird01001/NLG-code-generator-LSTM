
import datetime
def today_is_later_than(year, month, day):
    today = datetime.date.today()
    given = datetime.date(year, month, day)
    return today > given
