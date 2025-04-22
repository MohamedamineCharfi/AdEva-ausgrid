# energy_data/authentication.py
import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from mongoengine.errors import DoesNotExist
from energy_data.models import SuperUser

class MongoJWTAuthentication(authentication.BaseAuthentication):
    """
    Looks for Authorization: Bearer <token>,
    decodes with Django SECRET_KEY, then loads SuperUser by its ObjectId.
    """

    def authenticate(self, request):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ", 1)[1]
        print(f"Token received: {token}")  # DEBUG LOG

        try:
            # decode the token using the correct SECRET_KEY
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            print(f"Decoded payload: {payload}")  # DEBUG LOG
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Token has expired")
        except jwt.PyJWTError:
            raise exceptions.AuthenticationFailed("Invalid token")

        # Use username instead of user_id from the token payload
        username = payload.get("username")
        if not username:
            raise exceptions.AuthenticationFailed("Token payload missing username")

        try:
            user = SuperUser.objects.get(username=username)
        except DoesNotExist:
            raise exceptions.AuthenticationFailed("User not found")

        return (user, None)
        