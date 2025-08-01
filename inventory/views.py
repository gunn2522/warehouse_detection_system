from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpResponseServerError
from django.utils.timezone import now
from django.db.models import Sum
from django.http import StreamingHttpResponse
import csv

from .models import TruckLog, CylinderInventory
from .roboflow_utils import detect_and_classify_cylinders
from . import camera_detection

# Global state to track trucks across frames
last_seen_trucks = []
last_seen_cylinder_count = 0

def video_feed(request):
    stream_generator = camera_detection.start_camera_detection()
    if stream_generator is None:
        return HttpResponseServerError("Failed to connect to camera.")
    return StreamingHttpResponse(
        stream_generator,
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def live_detection_data(request):
    result = camera_detection.get_detection_result()
    return JsonResponse(result)

def dashboard_view(request):
    today = now().date()
    total_trucks = TruckLog.objects.filter(timestamp__date=today).count()
    total_loaded = TruckLog.objects.filter(timestamp__date=today, truck_type="loaded").aggregate(Sum('cylinder_count'))['cylinder_count__sum'] or 0
    total_unloaded = TruckLog.objects.filter(timestamp__date=today, truck_type="unloaded").aggregate(Sum('cylinder_count'))['cylinder_count__sum'] or 0

    # Get latest stock
    latest_inventory = CylinderInventory.objects.order_by('-last_updated').first()
    total_stock = latest_inventory.count if latest_inventory else 0

    context = {
        'total_trucks': total_trucks,
        'total_loaded': total_loaded,
        'total_unloaded': total_unloaded,
        'total_stock': total_stock
    }
    return render(request, 'dashboard.html', context)

def export_truck_logs(request):
    logs = TruckLog.objects.all().order_by('-timestamp')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="truck_logs.csv"'

    writer = csv.writer(response)
    writer.writerow(['Timestamp', 'Truck Type', 'Cylinder Count'])

    for log in logs:
        writer.writerow([log.timestamp, log.truck_type, log.cylinder_count])

    return response
