"""
temperature_sensor.py

Reads temperature from a DS18B20 sensor connected to Raspberry Pi via 1-Wire.
This module is designed to be imported into a main touchscreen UI script.
"""

import os
import glob
import time
import logging
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the 1-Wire kernel modules
os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

# Base path for 1-Wire device
BASE_DIR = '/sys/bus/w1/devices/'

try:
    DEVICE_PATH = glob.glob(BASE_DIR + '28*')[0]
    ROM = DEVICE_PATH.split('/')[-1]
    logger.info(f'Detected DS18B20 sensor: {ROM}')
except IndexError:
    logger.error("No DS18B20 sensor found under /sys/bus/w1/devices/")
    DEVICE_PATH = None
    ROM = None


def read_temp_raw() -> Optional[Tuple[str, str]]:
    """
    Read the raw sensor output from the w1_slave file.

    Returns:
        Tuple[str, str]: A tuple containing the validity line and temperature data line.
    """
    if not DEVICE_PATH:
        return None

    try:
        with open(DEVICE_PATH + '/w1_slave', 'r') as f:
            lines = f.readlines()
        return lines[0], lines[1]
    except Exception as e:
        logger.error(f"Error reading temperature sensor file: {e}")
        return None


def read_temp() -> Optional[Tuple[float, float]]:
    """
    Reads and parses the temperature from the sensor.

    Returns:
        Tuple[float, float]: The temperature in Celsius and Fahrenheit, or None if sensor not ready.
    """
    data = read_temp_raw()
    if data is None:
        return None

    valid, temp_line = data

    retry_count = 5
    while 'YES' not in valid and retry_count > 0:
        time.sleep(0.2)
        data = read_temp_raw()
        if data is None:
            return None
        valid, temp_line = data
        retry_count -= 1

    if 't=' in temp_line:
        try:
            temp_string = temp_line.split('t=')[1]
            temp_c = float(temp_string) / 1000.0
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            return round(temp_c, 2), round(temp_f, 2)
        except ValueError:
            logger.warning("Could not parse temperature value.")
            return None

    return None
