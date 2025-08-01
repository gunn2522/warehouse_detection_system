### STEP 3: Update `urls.py`

# inventory/urls.py
from django.urls import path,include

from . import views
from .views import dashboard_view, export_truck_logs,video_feed

urlpatterns = [
    path('dashboard/', dashboard_view, name='dashboard'),
    path('export-logs/', export_truck_logs, name='export_truck_logs'),
    path('video_feed/', views.video_feed, name='video_feed'),

    path('live_detection_data/', views.live_detection_data, name='live_detection_data'),
]