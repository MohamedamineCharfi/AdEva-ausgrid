from rest_framework import generics, status, exceptions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import render
from django.contrib.auth import authenticate
from rest_framework.permissions import AllowAny
from energy_data.models import Consumer, EnergyRecord, SuperUser
from .serializers import (
    ConsumerSerializer, 
    EnergyRecordSerializer,
    EnergyRecordBulkUploadSerializer,
    SuperUserSerializer,
    MyTokenObtainPairSerializer
)
from rest_framework_simplejwt.views import TokenObtainPairView
import pandas as pd
from dateutil.parser import parse
import io
from datetime import datetime

def create_consumer(consumer_id, postcode):
    collection = Consumer.get_collection()
    return collection.insert_one({
        'Customer': consumer_id,
        'Postcode': postcode
    })

def create_energy_record(record_data):
    collection = EnergyRecord.get_collection()
    if isinstance(record_data['date'], str):
        record_data['date'] = datetime.strptime(record_data['date'], '%m/%d/%Y')
    return collection.insert_one(record_data)

def dashboard_view(request):
    return render(request, 'dashboard.html')

class ConsumerListView(generics.ListCreateAPIView):
    serializer_class = ConsumerSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        collection = Consumer.get_collection()
        
        # Get unique consumers from the energy_records collection
        pipeline = [
            {"$group": {"_id": {"Customer": "$Customer", "Postcode": "$Postcode"}}},
            {"$project": {"Customer": "$_id.Customer", "Postcode": "$_id.Postcode", "_id": 0}}
        ]
        
        try:
            consumers = list(collection.aggregate(pipeline))
            print(f"Retrieved {len(consumers)} consumers from database")
            print(f"Sample data: {consumers[:2] if consumers else 'No data'}")
            return consumers
        except Exception as e:
            print(f"Error retrieving consumers: {e}")
            return []

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = create_consumer(
            consumer_id=serializer.validated_data['Customer'],
            postcode=serializer.validated_data['Postcode']
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class EnergyRecordListView(generics.ListCreateAPIView):
    serializer_class = EnergyRecordSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = EnergyRecord.objects   # a QuerySet
        customer = self.request.query_params.get('Customer')
        if customer:
            qs = qs.filter(Customer=int(customer))

        postcode = self.request.query_params.get('Postcode')
        if postcode:
            qs = qs.filter(Postcode=int(postcode))

        start_date = self.request.query_params.get('start_date')
        end_date   = self.request.query_params.get('end_date')
        if start_date:
            qs = qs.filter(date__gte=datetime.fromisoformat(start_date))
        if end_date:
            qs = qs.filter(date__lte=datetime.fromisoformat(end_date))

        return qs   # QuerySet is iterable of Document instances

    def list(self, request, *args, **kwargs):
        objs = self.get_queryset()
        # DRF’s Serializer will happily read attributes off each Document
        serializer = self.get_serializer(objs, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = create_energy_record(serializer.validated_data)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class EnergyRecordBulkUploadView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        serializer = EnergyRecordBulkUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            
            try:
                data = pd.read_csv(file)
                energy_records = []
                
                for _, row in data.iterrows():
                    # Create consumer if not exists
                    consumer_collection = Consumer.get_collection()
                    consumer_collection.update_one(
                        {'Customer': row['Customer']},
                        {'$setOnInsert': {
                            'Customer': row['Customer'],
                            'Postcode': row['Postcode']
                        }},
                        upsert=True
                    )
                    
                    # Prepare energy record
                    record = {
                        'Customer': row['Customer'],
                        'Postcode': row['Postcode'],
                        'date': parse(row['date']),
                        'consumption': row['consumption'],
                        'is_holiday_or_weekend': bool(row['is_holiday_or_weekend']),
                        'saison': row['saison'],
                        'consumption_daily_normalized': row['consumption_daily_normalized']
                    }
                    energy_records.append(record)
                
                # Bulk insert
                if energy_records:
                    EnergyRecord.get_collection().insert_many(energy_records)
                
                return Response({"message": "Data uploaded successfully"}, status=status.HTTP_201_CREATED)
            
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PredictEnergyConsumptionView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        try:
            input_data = request.data
            prediction = {
                'predicted_consumption': 0.5,
                'confidence': 0.95,
                'input_data': input_data
            }
            return Response(prediction, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class RegisterView(generics.CreateAPIView):
    serializer_class = SuperUserSerializer
    permission_classes = [AllowAny]

class MyTokenObtainPairView(TokenObtainPairView):
    permission_classes = []  # allow anyone to hit login
    def post(self, request):
        serializer = MyTokenObtainPairSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data, status=status.HTTP_200_OK)