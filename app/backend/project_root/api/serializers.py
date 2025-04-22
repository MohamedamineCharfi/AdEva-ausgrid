from rest_framework import serializers
from energy_data.models import Consumer, EnergyRecord
from energy_data.models import SuperUser
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth.hashers import make_password, check_password
from rest_framework_simplejwt.tokens import RefreshToken

class ConsumerSerializer(serializers.Serializer):
    # Update field names to match what's in your MongoDB documents
    Customer = serializers.IntegerField()
    Postcode = serializers.IntegerField()
    
    class Meta:
        fields = ['Customer', 'Postcode']

class EnergyRecordSerializer(serializers.Serializer):
    # Update field names to match what's in your MongoDB documents
    Customer = serializers.IntegerField()
    Postcode = serializers.IntegerField()
    date = serializers.DateTimeField()
    consumption = serializers.FloatField()
    is_holiday_or_weekend = serializers.BooleanField()
    saison = serializers.IntegerField()
    consumption_daily_normalized = serializers.FloatField()
    
    class Meta:
        fields = [
            'Customer', 'Postcode', 'date', 'consumption', 
            'is_holiday_or_weekend', 'saison',
            'consumption_daily_normalized'
        ]

class EnergyRecordBulkUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class SuperUserSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        return SuperUser(**validated_data).save()
    
class MyTokenObtainPairSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
    access = serializers.CharField(read_only=True)
    refresh = serializers.CharField(read_only=True)

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["username"] = user.username
        return token
    
    def validate(self, attrs):
        username = attrs.get("username")
        password = attrs.get("password")
        
        # Log only the username, NOT the password, for security purposes
        print(f"Attempting to log in user: {username}")

        # Fetch the user from the database based on username
        try:
            user = SuperUser.objects.get(username=username)
        except SuperUser.DoesNotExist:
            raise serializers.ValidationError("No account found with these credentials")

        # Verify the password
        if not check_password(password, user.password):
            raise serializers.ValidationError("Incorrect credentials")

        # Generate refresh and access tokens
        refresh = RefreshToken.for_user(user)

        # Return the tokens
        return {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }