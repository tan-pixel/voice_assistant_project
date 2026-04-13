import re

def parse_time_to_seconds(time_str):
    """Converts strings like '5 minutes' or '30 seconds' to integers."""
    if not time_str:
        return 0
        
    time_str = str(time_str).lower()
    total_seconds = 0
    
    # Extract numbers
    numbers = re.findall(r'\d+', time_str)
    if not numbers:
        return 0
        
    value = int(numbers[0])
    
    if 'hour' in time_str:
        total_seconds = value * 3600
    elif 'min' in time_str:
        total_seconds = value * 60
    elif 'sec' in time_str:
        total_seconds = value
    else:
        # Default to seconds if no unit is found
        total_seconds = value 
        
    return total_seconds