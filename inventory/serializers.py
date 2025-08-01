from rest_framework import serializers
from .models import CylinderInventory, TruckLog

class CylinderInventorySerializer(serializers.ModelSerializer):
    class Meta:
        model = CylinderInventory
        fields = '__all__'

class TruckLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TruckLog
        fields = '__all__'
