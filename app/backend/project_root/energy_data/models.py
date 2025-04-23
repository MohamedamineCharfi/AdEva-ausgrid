from django.contrib.auth.hashers import make_password, check_password
from mongoengine import Document, IntField, StringField, DateTimeField, FloatField


class Consumer(Document):
    # MongoEngine handles _id automatically
    Customer = IntField(unique=True)
    Postcode = IntField()

    # MongoEngine does not need 'Meta' for abstract models in this case
    meta = {
        'collection': 'consumers'  # Optional, you can specify collection name explicitly
    }

    @classmethod
    def get_collection(cls):
        # MongoEngine handles connections automatically
        return cls.objects  # MongoEngine query set

# For the EnergyRecord model using MongoEngine:
class EnergyRecord(Document):
    # MongoEngine handles _id automatically
    Customer = IntField()  # Just a regular field, not ForeignKey
    Postcode = IntField()
    date = DateTimeField()
    consumption = FloatField()
    is_holiday_or_weekend = IntField()
    saison = IntField()
    consumption_daily_normalized = FloatField()

    meta = {
        'collection': 'energy_records'  # Optional, specify collection name explicitly
    }

    @classmethod
    def get_collection(cls):
        return cls._get_collection()

class SuperUser(Document):
    meta = {"collection": "superusers"}  # optional, name of your MongoDB collection
    username = StringField(required=True, unique=True)
    password = StringField(required=True)

    @property
    def is_authenticated(self):
        return True
    
    def set_password(self, raw_password):
        self.password = make_password(raw_password)
        self.save()

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)