# middleware.py
import threading

_local = threading.local()

class SessionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # print(request.session['Email'])
        # Capture email from session at the start of each request
        _local.Email = request.session.get('Email',None)
        response = self.get_response(request)
        return response

    @staticmethod
    def get_email():
        """Retrieve the email from the thread-local storage."""
        return getattr(_local, 'Email', None)
