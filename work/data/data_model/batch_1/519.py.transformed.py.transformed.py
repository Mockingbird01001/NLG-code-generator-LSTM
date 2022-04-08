import calendar
import time
from email.utils import formatdate, parsedate, parsedate_tz
from datetime import datetime, timedelta
TIME_FMT = "%a, %d %b %Y %H:%M:%S GMT"
def expire_after(delta, date=None):
    date = date or datetime.utcnow()
    return date + delta
def datetime_to_header(dt):
    return formatdate(calendar.timegm(dt.timetuple()))
class BaseHeuristic(object):
    def warning(self, response):
        return '110 - "Response is Stale"'
    def update_headers(self, response):
        return {}
    def apply(self, response):
        updated_headers = self.update_headers(response)
        if updated_headers:
            response.headers.update(updated_headers)
            warning_header_value = self.warning(response)
            if warning_header_value is not None:
                response.headers.update({"Warning": warning_header_value})
        return response
class OneDayCache(BaseHeuristic):
    def update_headers(self, response):
        headers = {}
        if "expires" not in response.headers:
            date = parsedate(response.headers["date"])
            expires = expire_after(timedelta(days=1), date=datetime(*date[:6]))
            headers["expires"] = datetime_to_header(expires)
            headers["cache-control"] = "public"
        return headers
class ExpiresAfter(BaseHeuristic):
    def __init__(self, **kw):
        self.delta = timedelta(**kw)
    def update_headers(self, response):
        expires = expire_after(self.delta)
        return {"expires": datetime_to_header(expires), "cache-control": "public"}
    def warning(self, response):
        tmpl = "110 - Automatically cached for %s. Response might be stale"
        return tmpl % self.delta
class LastModified(BaseHeuristic):
    cacheable_by_default_statuses = {
        200, 203, 204, 206, 300, 301, 404, 405, 410, 414, 501
    }
    def update_headers(self, resp):
        headers = resp.headers
        if "expires" in headers:
            return {}
        if "cache-control" in headers and headers["cache-control"] != "public":
            return {}
        if resp.status not in self.cacheable_by_default_statuses:
            return {}
        if "date" not in headers or "last-modified" not in headers:
            return {}
        date = calendar.timegm(parsedate_tz(headers["date"]))
        last_modified = parsedate(headers["last-modified"])
        if date is None or last_modified is None:
            return {}
        now = time.time()
        current_age = max(0, now - date)
        delta = date - calendar.timegm(last_modified)
        freshness_lifetime = max(0, min(delta / 10, 24 * 3600))
        if freshness_lifetime <= current_age:
            return {}
        expires = date + freshness_lifetime
        return {"expires": time.strftime(TIME_FMT, time.gmtime(expires))}
    def warning(self, resp):
        return None
