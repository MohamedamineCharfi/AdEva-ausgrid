from django.urls import path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import (
    dashboard_view,
    RegisterView,
    ConsumerListView,
    EnergyRecordListView,
    EnergyRecordBulkUploadView,
    PredictConsumptionView,
    MyTokenObtainPairView  # Ensure this is imported
)

schema_view = get_schema_view(
    openapi.Info(
        title="Energy Consumption API",
        default_version='v1',
    description="API for energy consumption prediction system",
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

def health_check(request):
    return HttpResponse("OK")

urlpatterns = [
    # Frontend
    path('', dashboard_view, name='dashboard'),
    
    
    path('health/', health_check, name='health-check'),
    # Authentication
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', MyTokenObtainPairView.as_view(), name='login'),  # Use the custom view
    
    # API Endpoints
    path('consumers/', ConsumerListView.as_view(), name='consumer-list'),
    path('records/', EnergyRecordListView.as_view(), name='record-list'),
    path('upload/', EnergyRecordBulkUploadView.as_view(), name='bulk-upload'),
    path('predict/', PredictConsumptionView.as_view(), name='predict'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),  # Added token endpoint
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),  # Added refresh endpoint
    
    # Documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]