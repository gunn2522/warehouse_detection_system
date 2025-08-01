from django.contrib import admin
from .models import CylinderInventory, TruckLog
from django.db.models import Sum
from django.utils.html import format_html


@admin.register(CylinderInventory)
class CylinderInventoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'count', 'last_updated')
    ordering = ('-last_updated',)


@admin.register(TruckLog)
class TruckLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'truck_type', 'cylinder_count', 'remaining_cylinders', 'truck_total')
    list_filter = ('truck_type', 'timestamp')
    ordering = ('-timestamp',)
    search_fields = ('truck_type',)
    date_hierarchy = 'timestamp'
    readonly_fields = ('remaining_cylinders', 'truck_total')

    def remaining_cylinders(self, obj):
        inventory = CylinderInventory.objects.order_by('-last_updated').first()
        count = inventory.count if inventory else 0
        return count
    remaining_cylinders.short_description = "Cylinders Left in DB"

    def truck_total(self, obj):
        truck = obj.truck_type
        total = TruckLog.objects.filter(truck_type=truck).aggregate(
            Sum('cylinder_count')
        )['cylinder_count__sum'] or 0
        return total
    truck_total.short_description = "Total Cylinders for this Truck"
