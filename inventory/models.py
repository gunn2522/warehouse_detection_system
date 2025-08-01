from django.db import models
from django.utils import timezone

class CylinderInventory(models.Model):
    count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Cylinders Available: {self.count}"

class TruckLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    truck_type = models.CharField(max_length=20)
    cylinder_count = models.PositiveIntegerField()
    cylinders_left = models.IntegerField(default=0)
    total_cylinders = models.IntegerField(default=0)
    truck_identifier = models.CharField(max_length=50, blank=True, null=True)  # <- Add this


    def __str__(self):
        return f"{self.timestamp.date()} - {self.truck_type} - {self.cylinder_count} cylinders"


class Item:
    pass