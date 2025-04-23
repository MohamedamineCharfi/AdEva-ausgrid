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

class PredictConsumptionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Expects JSON payload:
            { "consumerId": 3, "month": "2013-06" }
        Returns:
            { "predicted_consumption": <float> }
        """
        consumer_id = request.data.get("consumerId")
        month_str  = request.data.get("month")     # e.g. "2013-06"

        # --- Validate inputs ---
        try:
            consumer_id = int(consumer_id)
        except (TypeError, ValueError):
            return Response(
                {"error": "Invalid or missing consumerId (must be integer)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not month_str or not isinstance(month_str, str):
            return Response(
                {"error": "Missing month. Use YYYY-MM format."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            year, mon = map(int, month_str.split("-"))
            start_date = datetime(year, mon, 1)
            # For end date, go to first of next month
            if mon == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, mon + 1, 1)
        except Exception:
            return Response(
                {"error": "month must be in YYYY-MM format."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- Fetch that month’s records for this consumer ---
        month_records = EnergyRecord.objects.filter(
            Customer=consumer_id,
            date__gte=start_date,
            date__lt=end_date
        )

        if not month_records:
            return Response(
                {"error": f"No data for consumer {consumer_id} in {month_str}."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- TODO: plug in your real ML/model here ---
        # For demo, we’ll just take the average daily consumption × 7:
        values = [r.consumption for r in month_records]
        avg_daily = sum(values) / len(values)
        predicted_week = avg_daily * 7

        return Response(
            {"predicted_consumption": round(predicted_week, 2)},
            status=status.HTTP_200_OK
        )
class RegisterView(generics.CreateAPIView):
    serializer_class = SuperUserSerializer
    permission_classes = [AllowAny]

class MyTokenObtainPairView(TokenObtainPairView):
    permission_classes = []  # allow anyone to hit login
    def post(self, request):
        serializer = MyTokenObtainPairSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data, status=status.HTTP_200_OK)