import csv
from django.core.management.base import BaseCommand
from datetime import datetime
from energy_app.models import Consumer, EnergyRecord

class Command(BaseCommand):
    help = 'Import energy data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            
            consumers_col = Consumer.get_collection()
            records_col = EnergyRecord.get_collection()
            
            for row in reader:
                try:
                    # Create or update consumer
                    consumer_id = int(row['Customer'])
                    postcode = int(row['Postcode'])
                    
                    consumers_col.update_one(
                        {'consumer_id': consumer_id},
                        {'$set': {'postcode': postcode}},
                        upsert=True
                    )
                    
                    # Parse date
                    date_str = row['date']
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    
                    # Create energy record
                    record_data = {
                        'consumer_id': consumer_id,
                        'date': date_obj,
                        'consumption': float(row['consumption']),
                        'is_holiday_or_weekend': bool(int(row['is_holiday_or_weekend'])),
                        'saison': int(row['saison']),
                        'dayofweek': date_obj.weekday(),
                        'dayofmonth': date_obj.day,
                        'monthofyear': date_obj.month,
                        'year': date_obj.year,
                        'consumption_daily_normalized': float(row['consumption_daily_normalized'])
                    }
                    
                    records_col.insert_one(record_data)
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f"Error processing row {row}: {str(e)}"
                    ))
                    continue
        
        self.stdout.write(self.style.SUCCESS(
            f"Successfully imported data from {csv_file}"
        ))