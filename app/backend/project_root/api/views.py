from rest_framework import generics, status, exceptions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import render
from django.contrib.auth import authenticate
from rest_framework.permissions import AllowAny
import torch
import os
from core.models import TimeSeriesTransformer
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

# load once at import time
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'transformer_energy_forecast_best.pt')

# These values must exactly match how you trained:
FEATURE_SIZE       = 3      # ['consumption_daily_normalized','is_holiday_or_weekend','saison']
D_MODEL            = 64
NHEAD              = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD    = 128
DROPOUT            = 0.1
FORECAST_HORIZON   = 7


MODEL = TimeSeriesTransformer(
    feature_size=FEATURE_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    forecast_horizon=FORECAST_HORIZON,
)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
MODEL.eval()


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
        Expects JSON payload: { "consumerId": 3, "month": "2013-06" }
        Returns: {"predicted_consumption": [<float>, …, <float>]} 7 values
        """
        # --- Validate & parse inputs ---
        try:
            consumer_id = int(request.data.get("consumerId"))
        except (TypeError, ValueError):
            return Response(
                {"error": "Invalid or missing consumerId (must be integer)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        month_str = request.data.get("month")
        if not month_str or not isinstance(month_str, str):
            return Response(
                {"error": "Missing month. Use YYYY-MM format."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            year, mon = map(int, month_str.split("-"))
            start_date = datetime(year, mon, 1)
            end_date = datetime(year + (mon == 12), (mon % 12) + 1, 1)
        except Exception:
            return Response(
                {"error": "month must be in YYYY-MM format."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- Fetch that month’s records for this consumer, ordered by date ---
        month_records = EnergyRecord.objects.filter(
            Customer=consumer_id,
            date__gte=start_date,
            date__lt=end_date
        ).order_by('date')

        if not month_records:
            return Response(
                {"error": f"No data for consumer {consumer_id} in {month_str}."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- Build the input tensor (last 30 days) ---
        values = []
        for r in month_records:
            # Example: Assuming r.consumption, r.is_holiday_or_weekend, r.saison are available
            consumption = r.consumption
            is_holiday_or_weekend = 1 if r.is_holiday_or_weekend else 0  # 1 or 0
            saison = r.saison  # Assuming it's already a numerical value
            values.append([consumption, is_holiday_or_weekend, saison])

        if len(values) < 30:
            return Response(
                {"error": "Insufficient data (need at least 30 daily records)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Ensure tensor is of shape (1, 30, 3) where 30 = seq_len, 3 = features
        input_tensor = torch.tensor([values[-30:]], dtype=torch.float32)

        # --- Run inference ---
        with torch.no_grad():
            output = MODEL(input_tensor)  # Assuming MODEL expects shape (1, 30, 3)
            weekly_forecast = output.squeeze(0).tolist()

        return Response(
            {"predicted_consumption": weekly_forecast},
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