#!/usr/bin/env python3
r"""
Drone Flight Context Map Generator - Enhanced with GPS Coordinate Display
Generates contextual maps showing drone flight areas with surrounding geography and GPS coordinates.
Run with: C:\OSGeo4W\bin\python-qgis.bat drone_map_generator.py
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import math
import tempfile
import logging
from datetime import datetime

# QGIS imports - requires QGIS installation with PyQGIS
try:
    from qgis.core import (
        QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
        QgsPointXY, QgsRectangle, QgsCoordinateReferenceSystem,
        QgsMapSettings, QgsMapRendererCustomPainterJob, QgsMapRendererSequentialJob,
        QgsFeature, QgsGeometry, QgsSymbol, QgsRendererRange, QgsGraduatedSymbolRenderer,
        QgsMarkerSymbol, QgsSimpleMarkerSymbolLayer, QgsLineSymbol, QgsField,
        QgsFields, QgsWkbTypes, QgsFillSymbol, QgsProject, QgsLayerTreeLayer,
        QgsApplication, QgsProcessingFeedback, QgsVectorFileWriter, QgsPalLayerSettings,
        QgsTextFormat, QgsVectorLayerSimpleLabeling
    )
    from qgis.PyQt.QtCore import QSize, QSizeF, QVariant
    from qgis.PyQt.QtGui import QColor, QPainter, QImage, QFont
    from qgis.PyQt.QtCore import Qt
    QGIS_AVAILABLE = True
    print("✅ PyQGIS imported successfully")
except ImportError as e:
    QGIS_AVAILABLE = False
    print(f"❌ PyQGIS not available: {e}")

# =============================================================================
# 🔧 CONFIGURATION - CHANGE THESE FOR YOUR NEEDS
# =============================================================================

# Output settings
OUTPUT_FOLDER = "/Users/abhivur/Documents/SMU_REU/project/HawkEye/path"  # Where to save generated maps
MAP_WIDTH = 2048   # Output image width in pixels
MAP_HEIGHT = 1536  # Output image height in pixels
MAP_DPI = 300      # High DPI for quality

# Context settings - how much area around flight path to show
CONTEXT_BUFFER_PERCENT = 0.3  # Add 30% buffer around flight area (0.5 = 50% more area)
MIN_MAP_SIZE_METERS = 1000    # Minimum map dimension in meters (for very small flight areas)

# Map layer settings
DEFAULT_BASE_LAYER = "OpenStreetMap"  # Options: "OpenStreetMap", "Google Satellite", "Bing Aerial"
INCLUDE_TERRAIN = True       # Add terrain/elevation layer
INCLUDE_LABELS = True        # Include place name labels

# Flight path visualization
FLIGHT_PATH_COLOR = "red"    # Color for flight path line
FLIGHT_PATH_WIDTH = 0.5        # Line width for flight path
WAYPOINT_COLOR = "blue"      # Color for waypoint markers
WAYPOINT_SIZE = 8            # Size of waypoint markers

# GPS Coordinate display settings - SIMPLIFIED OUTPUT
SHOW_GPS_LABELS = False      # Don't show GPS coordinates as labels on map
GPS_LABEL_SIZE = 8           # Font size for GPS labels
GPS_LABEL_COLOR = "black"    # Color for GPS labels
EXPORT_GPS_CSV = True        # Export GPS coordinates to CSV file
GPS_PRECISION = 6            # Decimal places for GPS coordinates

# Console output control
VERBOSE_CONSOLE_OUTPUT = False  # Disable detailed console output
EXPORT_JSON_SUMMARY = False     # Don't export JSON summary

# Supported input formats
SUPPORTED_FORMATS = ['.gpx', '.kml', '.kmz', '.csv', '.json', '.txt']

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drone_map_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# FLIGHT MAP DATA CLASSES
# =============================================================================

class FlightPoint:
    """Represents a single point in a flight plan"""
    def __init__(self, lat: float, lon: float, alt: Optional[float] = None, 
                 name: str = "", timestamp: Optional[str] = None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.name = name
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"FlightPoint(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt})"
    
    def get_gps_string(self, precision: int = GPS_PRECISION) -> str:
        """Get formatted GPS coordinate string"""
        return f"{self.lat:.{precision}f}, {self.lon:.{precision}f}"
    
    def get_detailed_info(self) -> Dict[str, str]:
        """Get detailed information about this waypoint"""
        info = {
            'Name': self.name or "Unnamed",
            'Latitude': f"{self.lat:.{GPS_PRECISION}f}",
            'Longitude': f"{self.lon:.{GPS_PRECISION}f}",
            'GPS Coordinates': self.get_gps_string(),
        }
        
        if self.alt is not None:
            info['Altitude'] = f"{self.alt:.1f}m"
        
        if self.timestamp:
            info['Timestamp'] = self.timestamp
            
        return info

class FlightPlan:
    """Represents a complete flight plan with waypoints and metadata"""
    def __init__(self, points: List[FlightPoint], name: str = "Flight Plan", 
                 description: str = ""):
        self.points = points
        self.name = name
        self.description = description
        self.bounds = self._calculate_bounds()
        self.created_at = datetime.now().isoformat()
    
    def _calculate_bounds(self) -> Dict[str, float]:
        """Calculate the bounding box of all flight points"""
        if not self.points:
            return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}
        
        lats = [p.lat for p in self.points]
        lons = [p.lon for p in self.points]
        
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }
    
    def get_center_point(self) -> Tuple[float, float]:
        """Get the center point of the flight area"""
        bounds = self.bounds
        center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
        center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
        return center_lat, center_lon
    
    def get_area_size_meters(self) -> Tuple[float, float]:
        """Calculate the approximate flight area size in meters"""
        bounds = self.bounds
        
        # Use Haversine formula for more accurate distance calculation
        lat_diff_m = haversine_distance(
            bounds["min_lat"], bounds["min_lon"],
            bounds["max_lat"], bounds["min_lon"]
        )
        lon_diff_m = haversine_distance(
            bounds["min_lat"], bounds["min_lon"],
            bounds["min_lat"], bounds["max_lon"]
        )
        
        return lat_diff_m, lon_diff_m
    
    def print_gps_coordinates(self):
        """Print GPS coordinates to console - simplified output"""
        if not VERBOSE_CONSOLE_OUTPUT:
            print(f"📍 Extracted {len(self.points)} GPS coordinates from {self.name}")
            return
            
        print(f"\n📍 GPS Coordinates for Flight Plan: {self.name}")
        print("=" * 60)
        
        for i, point in enumerate(self.points, 1):
            info = point.get_detailed_info()
            print(f"Waypoint {i}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()
    
    def export_gps_to_csv(self, output_folder: str = OUTPUT_FOLDER) -> str:
        """Export GPS coordinates to CSV file"""

        # Override the output folder with your hardcoded path
        output_folder = "/Users/abhivur/Documents/SMU_REU/project/HawkEye/path"
        
        csv_filename = f"{self.name}_gps_coordinates.csv"
        csv_path = Path(output_folder) / csv_filename

        # Prepare data for CSV
        data = []
        for i, point in enumerate(self.points, 1):
            row = {
                'Waypoint_Number': i,
                'Name': point.name or f"WP{i}",
                'Latitude': point.lat,
                'Longitude': point.lon,
                'GPS_Coordinates': point.get_gps_string(),
            }

            if point.alt is not None:
                row['Altitude_m'] = point.alt

            if point.timestamp:
                row['Timestamp'] = point.timestamp

            data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(data)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        logger.info(f"📊 GPS coordinates exported to: {csv_path}")
        return str(csv_path)

    def get_flight_summary(self) -> Dict[str, any]:
        """Get summary statistics for the flight plan"""
        if not self.points:
            return {}
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(self.points)):
            total_distance += haversine_distance(
                self.points[i-1].lat, self.points[i-1].lon,
                self.points[i].lat, self.points[i].lon
            )
        
        # Get altitude statistics
        altitudes = [p.alt for p in self.points if p.alt is not None]
        
        center_lat, center_lon = self.get_center_point()
        area_height, area_width = self.get_area_size_meters()
        
        summary = {
            'flight_plan_name': self.name,
            'total_waypoints': len(self.points),
            'total_distance_m': total_distance,
            'center_coordinates': f"{center_lat:.{GPS_PRECISION}f}, {center_lon:.{GPS_PRECISION}f}",
            'area_width_m': area_width,
            'area_height_m': area_height,
            'bounding_box': {
                'north': f"{self.bounds['max_lat']:.{GPS_PRECISION}f}",
                'south': f"{self.bounds['min_lat']:.{GPS_PRECISION}f}",
                'east': f"{self.bounds['max_lon']:.{GPS_PRECISION}f}",
                'west': f"{self.bounds['min_lon']:.{GPS_PRECISION}f}"
            }
        }
        
        if altitudes:
            summary['min_altitude_m'] = min(altitudes)
            summary['max_altitude_m'] = max(altitudes)
            summary['avg_altitude_m'] = sum(altitudes) / len(altitudes)
        
        return summary

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in meters"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def meters_to_degrees(meters: float, latitude: float) -> Tuple[float, float]:
    """Convert meters to degrees at given latitude"""
    # 1 degree latitude ≈ 111,000 meters
    lat_deg = meters / 111000
    
    # 1 degree longitude varies by latitude
    lon_deg = meters / (111000 * math.cos(math.radians(latitude)))
    
    return lat_deg, lon_deg

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def format_coordinates_dms(lat: float, lon: float) -> str:
    """Convert decimal degrees to degrees, minutes, seconds format"""
    def dd_to_dms(dd: float, is_latitude: bool) -> str:
        direction = ""
        if is_latitude:
            direction = "N" if dd >= 0 else "S"
        else:
            direction = "E" if dd >= 0 else "W"
        
        dd = abs(dd)
        degrees = int(dd)
        minutes_float = (dd - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        
        return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"
    
    lat_dms = dd_to_dms(lat, True)
    lon_dms = dd_to_dms(lon, False)
    
    return f"{lat_dms}, {lon_dms}"

# =============================================================================
# FLIGHT PLAN PARSERS (Enhanced with better coordinate handling)
# =============================================================================

class FlightPlanParser:
    """Parse various flight plan file formats into FlightPlan objects"""
    
    @staticmethod
    def parse_file(file_path: str) -> FlightPlan:
        """Auto-detect format and parse flight plan file"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        logger.info(f"📁 Parsing flight plan: {file_path.name}")
        logger.info(f"📄 Detected format: {extension}")
        
        if extension == '.gpx':
            return FlightPlanParser.parse_gpx(file_path)
        elif extension in ['.kml', '.kmz']:
            return FlightPlanParser.parse_kml(file_path)
        elif extension == '.csv':
            return FlightPlanParser.parse_csv(file_path)
        elif extension == '.json':
            return FlightPlanParser.parse_json(file_path)
        elif extension == '.txt':
            return FlightPlanParser.parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    @staticmethod
    def parse_gpx(file_path: Path) -> FlightPlan:
        """Parse GPX file format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle GPX namespace
        namespace = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        if not root.tag.endswith('gpx'):
            namespace = {}  # Try without namespace
        
        points = []
        
        # Parse waypoints
        for wpt in root.findall('.//waypoint' if not namespace else './/gpx:wpt', namespace):
            lat = float(wpt.get('lat'))
            lon = float(wpt.get('lon'))
            name = wpt.find('.//name' if not namespace else './/gpx:name', namespace)
            name = name.text if name is not None else ""
            
            # Try to get elevation
            ele = wpt.find('.//ele' if not namespace else './/gpx:ele', namespace)
            alt = float(ele.text) if ele is not None else None
            
            points.append(FlightPoint(lat=lat, lon=lon, alt=alt, name=name))
        
        # Parse track points if no waypoints found
        if not points:
            for trkpt in root.findall('.//trkpt' if not namespace else './/gpx:trkpt', namespace):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                
                # Try to get elevation
                ele = trkpt.find('.//ele' if not namespace else './/gpx:ele', namespace)
                alt = float(ele.text) if ele is not None else None
                
                points.append(FlightPoint(lat=lat, lon=lon, alt=alt))
        
        return FlightPlan(points=points, name=file_path.stem)
    
    @staticmethod
    def parse_kml(file_path: Path) -> FlightPlan:
        """Parse KML file format - handles LineString coordinates"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle KML namespace
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        if not root.tag.endswith('kml'):
            namespace = {}
        
        points = []
        
        # Find coordinates in LineString or Point elements
        for coordinates in root.findall('.//coordinates' if not namespace else './/kml:coordinates', namespace):
            coord_text = coordinates.text.strip() if coordinates.text else ""
            
            logger.info(f"🔍 Found coordinates block with {len(coord_text)} characters")
            
            # For LineString coordinates: "lon,lat,alt lon,lat,alt lon,lat,alt ..."
            # Split by spaces first to get individual coordinate triplets
            coord_triplets = coord_text.split()
            
            logger.info(f"📊 Found {len(coord_triplets)} coordinate triplets")
            
            for i, triplet in enumerate(coord_triplets):
                triplet = triplet.strip()
                if triplet:
                    try:
                        # Now split each triplet by commas
                        coords = triplet.split(',')
                        if len(coords) >= 2:
                            lon = float(coords[0])
                            lat = float(coords[1])
                            alt = float(coords[2]) if len(coords) > 2 and coords[2] else None
                            
                            points.append(FlightPoint(lat=lat, lon=lon, alt=alt))
                            
                            if i < 3:  # Log first few points for verification
                                logger.info(f"✅ Point {i+1}: lat={lat:.6f}, lon={lon:.6f}, alt={alt}")
                    
                    except (ValueError, IndexError) as e:
                        logger.warning(f"⚠️  Skipping invalid coordinate triplet: {triplet[:30]} - {e}")
        
        logger.info(f"📍 Successfully parsed {len(points)} waypoints from KML")
        
        if not points:
            raise ValueError("No valid coordinates found in KML file")
        
        return FlightPlan(points=points, name=file_path.stem)
    
    @staticmethod
    def parse_csv(file_path: Path) -> FlightPlan:
        """Parse CSV file with lat/lon columns"""
        df = pd.read_csv(file_path)
        
        # Try to find latitude and longitude columns (case insensitive)
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['lat', 'latitude', 'y']:
                lat_col = col
            elif col_lower in ['lon', 'lng', 'longitude', 'x']:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            raise ValueError("CSV must contain latitude and longitude columns")
        
        points = []
        for _, row in df.iterrows():
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            
            # Try to get altitude and name if available
            alt = None
            name = ""
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['alt', 'altitude', 'elevation', 'z']:
                    alt = float(row[col]) if pd.notna(row[col]) else None
                elif col_lower in ['name', 'waypoint', 'description']:
                    name = str(row[col]) if pd.notna(row[col]) else ""
            
            points.append(FlightPoint(lat=lat, lon=lon, alt=alt, name=name))
        
        return FlightPlan(points=points, name=file_path.stem)
    
    @staticmethod
    def parse_json(file_path: Path) -> FlightPlan:
        """Parse JSON file with flight data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        points = []
        
        # Handle different JSON structures
        if 'waypoints' in data:
            waypoints = data['waypoints']
        elif 'points' in data:
            waypoints = data['points']
        elif isinstance(data, list):
            waypoints = data
        else:
            raise ValueError("JSON format not recognized")
        
        for wp in waypoints:
            if isinstance(wp, dict):
                lat = wp.get('lat') or wp.get('latitude')
                lon = wp.get('lon') or wp.get('lng') or wp.get('longitude')
                alt = wp.get('alt') or wp.get('altitude')
                name = wp.get('name', '')
                
                if lat is not None and lon is not None:
                    points.append(FlightPoint(lat=float(lat), lon=float(lon), 
                                            alt=float(alt) if alt else None, name=str(name)))
        
        plan_name = data.get('name', file_path.stem)
        return FlightPlan(points=points, name=plan_name)
    
    @staticmethod
    def parse_txt(file_path: Path) -> FlightPlan:
        """Parse simple text file with lat,lon pairs"""
        points = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments
                    try:
                        coords = line.replace('\t', ',').split(',')
                        if len(coords) >= 2:
                            lat = float(coords[0].strip())
                            lon = float(coords[1].strip())
                            alt = float(coords[2].strip()) if len(coords) > 2 else None
                            points.append(FlightPoint(lat=lat, lon=lon, alt=alt))
                    except ValueError:
                        logger.warning(f"Skipping invalid line {line_num}: {line}")
        
        return FlightPlan(points=points, name=file_path.stem)

# =============================================================================
# QGIS MAP GENERATOR (Enhanced with GPS coordinate labels)
# =============================================================================

class QGISMapGenerator:
    """Generate contextual maps using QGIS with GPS coordinate display"""
    
    def _add_base_layers(self):
        """Add OpenStreetMap base map using XYZ tiles"""
        osm_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        uri = f"type=xyz&url={osm_url}&zmin=0&zmax=19"
        self.osm_layer = QgsRasterLayer(uri, "OpenStreetMap", "wms")  # sometimes "wms" works better than "xyz"

        if self.osm_layer.isValid():
            QgsProject.instance().addMapLayer(self.osm_layer)
            logger.info("✅ Added OpenStreetMap base layer")
        else:
            logger.warning("❌ Failed to add OpenStreetMap layer")



    def __init__(self):
        if not QGIS_AVAILABLE:
            raise ImportError("PyQGIS not available. Run with python-qgis.bat")
        
        # Initialize QGIS Application - Windows OSGeo4W path
        QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()
        
        # Create project
        self.project = QgsProject.instance()
        self.project.clear()
        
        logger.info("✅ QGIS initialized successfully")
    
    def generate_contextual_map(self, flight_plan: FlightPlan, 
                              output_filename: str = None) -> str:
        """Generate a contextual map for the flight plan with GPS coordinates"""
        
        if not flight_plan.points:
            raise ValueError("Flight plan has no points")
        
        logger.info(f"🗺️  Generating contextual map for flight plan: {flight_plan.name}")
        logger.info(f"📍 Flight area: {len(flight_plan.points)} waypoints")
        
        # Print GPS coordinates to console
        flight_plan.print_gps_coordinates()
        
        # Export GPS coordinates to CSV if requested
        if EXPORT_GPS_CSV:
            flight_plan.export_gps_to_csv()
        
        # Calculate map extent with context buffer
        map_extent = self._calculate_map_extent(flight_plan)
        logger.info(f"🔍 Map extent: {map_extent}")
        
        # Setup base layers
        self._add_base_layers()
        
        # Add flight plan layers with GPS labels
        self._add_flight_plan_layers(flight_plan)
        
        # Generate and save map
        if output_filename is None:
            output_filename = f"{flight_plan.name}_contextual_map.png"
        
        output_path = Path(OUTPUT_FOLDER) / output_filename
        self._render_map(map_extent, output_path)
        
        # Generate flight summary only if requested
        if EXPORT_JSON_SUMMARY:
            summary = flight_plan.get_flight_summary()
            self._save_flight_summary(summary, flight_plan.name)
        
        logger.info(f"✅ Map saved: {output_path}")
        print(f"✅ Generated: {output_path.name}")
        return str(output_path)
    
    def _calculate_map_extent(self, flight_plan: FlightPlan) -> QgsRectangle:
        """Calculate the map extent with context buffer"""
        bounds = flight_plan.bounds
        
        # Get center point and area size
        center_lat, center_lon = flight_plan.get_center_point()
        area_height_m, area_width_m = flight_plan.get_area_size_meters()
        
        # Ensure minimum map size
        area_height_m = max(area_height_m, MIN_MAP_SIZE_METERS)
        area_width_m = max(area_width_m, MIN_MAP_SIZE_METERS)
        
        # Add context buffer
        buffered_height_m = area_height_m * (1 + CONTEXT_BUFFER_PERCENT)
        buffered_width_m = area_width_m * (1 + CONTEXT_BUFFER_PERCENT)
        
        # Convert to degrees
        lat_buffer, lon_buffer = meters_to_degrees(
            max(buffered_height_m, buffered_width_m) / 2, center_lat
        )
        
        # Create extent rectangle
        min_x = center_lon - lon_buffer
        max_x = center_lon + lon_buffer
        min_y = center_lat - lat_buffer
        max_y = center_lat + lat_buffer
        
        return QgsRectangle(min_x, min_y, max_x, max_y)
    


    
    def _add_flight_plan_layers(self, flight_plan: FlightPlan):
        """Add flight plan waypoints and path as map layers with GPS coordinate labels"""
        
        # Create waypoints layer with attribute fields
        try:
            fields = QgsFields()
            # Fix deprecation warnings by using proper QgsField constructor
            fields.append(QgsField("id", QVariant.Int))
            fields.append(QgsField("name", QVariant.String, "text"))
            fields.append(QgsField("latitude", QVariant.Double, "double"))
            fields.append(QgsField("longitude", QVariant.Double, "double"))
            fields.append(QgsField("gps_coords", QVariant.String, "text"))
            fields.append(QgsField("altitude", QVariant.Double, "double"))
        except:
            # Fallback for older QGIS versions
            fields = QgsFields()
            fields.append(QgsField("id", QVariant.Int))
            fields.append(QgsField("name", QVariant.String))
            fields.append(QgsField("latitude", QVariant.Double))
            fields.append(QgsField("longitude", QVariant.Double))
            fields.append(QgsField("gps_coords", QVariant.String))
            fields.append(QgsField("altitude", QVariant.Double))
        
        waypoints_layer = QgsVectorLayer("Point?crs=EPSG:4326", "Waypoints", "memory")
        waypoints_provider = waypoints_layer.dataProvider()
        waypoints_provider.addAttributes(fields)
        waypoints_layer.updateFields()
        
        # Add waypoint features with GPS coordinate data
        features = []
        for i, point in enumerate(flight_plan.points):
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point.lon, point.lat)))
            
            attributes = [
                i + 1,
                point.name or f"WP{i+1}",
                point.lat,
                point.lon,
                point.get_gps_string(),
                point.alt or 0
            ]
            feature.setAttributes(attributes)
            features.append(feature)
        
        waypoints_provider.addFeatures(features)
        waypoints_layer.updateExtents()
        
        # Style waypoints
        symbol = QgsMarkerSymbol.createSimple({
            'name': 'circle',
            'color': WAYPOINT_COLOR,
            'size': str(WAYPOINT_SIZE),
            'outline_color': 'white',
            'outline_width': '1'
        })
        waypoints_layer.renderer().setSymbol(symbol)
        
        # Add GPS coordinate labels if requested
        if SHOW_GPS_LABELS:
            self._add_gps_labels(waypoints_layer)
        
        self.project.addMapLayer(waypoints_layer)
        logger.info(f"✅ Added waypoints layer with {len(flight_plan.points)} points")
        
        # Create flight path layer if multiple points
        if len(flight_plan.points) > 1:
            path_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Flight Path", "memory")
            path_provider = path_layer.dataProvider()
            
            # Create path geometry
            points = [QgsPointXY(p.lon, p.lat) for p in flight_plan.points]
            path_feature = QgsFeature()
            path_feature.setGeometry(QgsGeometry.fromPolylineXY(points))
            
            path_provider.addFeature(path_feature)
            path_layer.updateExtents()
            
            # Style flight path
            symbol = QgsLineSymbol.createSimple({
                'color': FLIGHT_PATH_COLOR,
                'width': str(FLIGHT_PATH_WIDTH),
                'capstyle': 'round'
            })
            path_layer.renderer().setSymbol(symbol)
            
            self.project.addMapLayer(path_layer)
            logger.info("✅ Added flight path layer")
    
    def _add_gps_labels(self, layer: QgsVectorLayer):
        """Add GPS coordinate labels to waypoint layer"""
        try:
            # Configure label settings
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = "gps_coords"
            label_settings.enabled = True
            
            # Configure text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", GPS_LABEL_SIZE))
            text_format.setSize(GPS_LABEL_SIZE)
            text_format.setColor(QColor(GPS_LABEL_COLOR))
            
            # Add background for better readability
            buffer = text_format.buffer()
            buffer.setEnabled(True)
            buffer.setSize(1)
            buffer.setColor(QColor("white"))
            text_format.setBuffer(buffer)
            
            label_settings.setFormat(text_format)
            
            # Position labels - fix for QGIS version compatibility
            try:
                # Try new enum style first (QGIS 3.x)
                from qgis.core import QgsPalLayerSettings
                label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
                label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove
            except (AttributeError, TypeError):
                # Fallback to older enum style
                try:
                    label_settings.placement = QgsPalLayerSettings.OverPoint
                    label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove
                except (AttributeError, TypeError):
                    # If all else fails, use basic positioning
                    label_settings.placement = 0  # OverPoint
                    label_settings.quadOffset = 0  # QuadrantAbove
            
            label_settings.yOffset = 2
            
            # Apply labeling
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
            
            logger.info("✅ Added GPS coordinate labels")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not add GPS labels: {e}")
            logger.info("📍 Map will be generated without coordinate labels")
    
    def _render_map(self, extent: QgsRectangle, output_path: Path):
        """Render the map to an image file"""
        
        # Setup map settings
        settings = QgsMapSettings()
        settings.setExtent(extent)
        settings.setOutputSize(QSize(MAP_WIDTH, MAP_HEIGHT))
        settings.setOutputDpi(MAP_DPI)
        settings.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        
        # Get all layers in correct order
        layers = [layer for layer in self.project.mapLayers().values()]
        settings.setLayers(layers)
        
        # Create image
        image = QImage(QSize(MAP_WIDTH, MAP_HEIGHT), QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.white)
        
        # Render map
        painter = QPainter(image)
        render = QgsMapRendererCustomPainterJob(settings, painter)
        render.start()
        render.waitForFinished()
        painter.end()
        
        # Save image
        ensure_output_directory()
        image.save(str(output_path))
        logger.info(f"🖼️  Map rendered: {MAP_WIDTH}x{MAP_HEIGHT} @ {MAP_DPI} DPI")
    
    def _save_flight_summary(self, summary: Dict[str, any], flight_name: str):
        """Save flight plan summary to JSON file"""
        summary_filename = f"{flight_name}_flight_summary.json"
        summary_path = Path(OUTPUT_FOLDER) / summary_filename
        
        ensure_output_directory()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Flight summary saved: {summary_path}")
        
        # Also print summary to console
        print(f"\n📊 Flight Plan Summary: {flight_name}")
        print("=" * 50)
        print(f"Total Waypoints: {summary['total_waypoints']}")
        print(f"Total Distance: {summary['total_distance_m']:.1f} meters")
        print(f"Center Coordinates: {summary['center_coordinates']}")
        print(f"Area Size: {summary['area_width_m']:.1f}m × {summary['area_height_m']:.1f}m")
        
        if 'min_altitude_m' in summary:
            print(f"Altitude Range: {summary['min_altitude_m']:.1f}m - {summary['max_altitude_m']:.1f}m")
            print(f"Average Altitude: {summary['avg_altitude_m']:.1f}m")
        
        print("\nBounding Box:")
        bbox = summary['bounding_box']
        print(f"  North: {bbox['north']}")
        print(f"  South: {bbox['south']}")
        print(f"  East: {bbox['east']}")
        print(f"  West: {bbox['west']}")
    
    def cleanup(self):
        """Clean up QGIS resources"""
        self.qgs.exitQgis()

# =============================================================================
# ENHANCED MAIN INTERFACE FUNCTIONS
# =============================================================================

def generate_flight_context_map(flight_plan_file: str, 
                               output_filename: str = None,
                               show_coordinates: bool = True,
                               export_coordinates: bool = True) -> str:
    """
    Main function to generate contextual map from flight plan file with GPS coordinates
    
    Args:
        flight_plan_file: Path to flight plan file (GPX, KML, CSV, JSON, TXT)
        output_filename: Optional custom output filename
        show_coordinates: Whether to display GPS coordinates as labels on map
        export_coordinates: Whether to export GPS coordinates to CSV
    
    Returns:
        Path to generated map image
    """
    
    logger.info("🚁 Starting Flight Context Map Generation with GPS Coordinates")
    logger.info("=" * 70)
    
    try:
        # Parse flight plan
        flight_plan = FlightPlanParser.parse_file(flight_plan_file)
        logger.info(f"✅ Parsed flight plan: {len(flight_plan.points)} waypoints")
        
        # Override global settings if specified
        global SHOW_GPS_LABELS, EXPORT_GPS_CSV
        SHOW_GPS_LABELS = show_coordinates
        EXPORT_GPS_CSV = export_coordinates
        
        # Generate map
        map_generator = QGISMapGenerator()
        
        try:
            output_path = map_generator.generate_contextual_map(
                flight_plan, output_filename
            )
            
            logger.info("🎉 SUCCESS! Contextual map with GPS coordinates generated")
            logger.info(f"📁 Output: {output_path}")
            
            return output_path
            
        finally:
            map_generator.cleanup()
    
    except Exception as e:
        logger.error(f"❌ Error generating map: {e}")
        raise

def extract_gps_coordinates_only(flight_plan_file: str) -> List[Dict[str, str]]:
    """
    Extract just the GPS coordinates from a flight plan file
    
    Args:
        flight_plan_file: Path to flight plan file
    
    Returns:
        List of dictionaries containing GPS coordinate information
    """
    
    try:
        flight_plan = FlightPlanParser.parse_file(flight_plan_file)
        
        coordinates = []
        for i, point in enumerate(flight_plan.points, 1):
            coord_info = {
                'waypoint_number': i,
                'name': point.name or f"WP{i}",
                'latitude': f"{point.lat:.{GPS_PRECISION}f}",
                'longitude': f"{point.lon:.{GPS_PRECISION}f}",
                'gps_coordinates': point.get_gps_string(),
                'dms_coordinates': format_coordinates_dms(point.lat, point.lon)
            }
            
            if point.alt is not None:
                coord_info['altitude_m'] = f"{point.alt:.1f}"
            
            coordinates.append(coord_info)
        
        return coordinates
    
    except Exception as e:
        logger.error(f"❌ Error extracting coordinates: {e}")
        raise

def print_gps_coordinates_table(flight_plan_file: str):
    """Print GPS coordinates in a formatted table - simplified"""
    
    coordinates = extract_gps_coordinates_only(flight_plan_file)
    
    if not VERBOSE_CONSOLE_OUTPUT:
        print(f"📊 {len(coordinates)} waypoints extracted")
        return
    
    print(f"\n📍 GPS Coordinates from {Path(flight_plan_file).name}")
    print("=" * 80)
    print(f"{'WP#':<4} {'Name':<12} {'Latitude':<12} {'Longitude':<13} {'Altitude':<10}")
    print("-" * 80)
    
    for coord in coordinates:
        wp_num = coord['waypoint_number']
        name = coord['name'][:11]  # Truncate long names
        lat = coord['latitude']
        lon = coord['longitude']
        alt = coord.get('altitude_m', 'N/A')
        
        print(f"{wp_num:<4} {name:<12} {lat:<12} {lon:<13} {alt:<10}")
    
    print("-" * 80)
    print(f"Total waypoints: {len(coordinates)}")


def test_with_user_data():
    """Test the enhanced map generator with user's drone flight plan - minimal output"""

    print("🗺️  Generating drone flight map and GPS coordinates...")

    # Directory containing flight plan file(s)
    user_folder = Path("/Users/abhivur/Documents/SMU_REU/project/HawkEye/path")

    if not user_folder.exists() or not user_folder.is_dir():
        print(f"❌ Folder not found or is not a directory: {user_folder}")
        print("💡 Please check the folder path")
        return

    # Look for a valid flight plan file inside the folder
    valid_extensions = [".kml", ".json", ".txt"]
    matching_files = [f for f in user_folder.iterdir() if f.is_file() and f.suffix.lower() in [ext.lower() for ext in valid_extensions]]

    if not matching_files:
        print(f"❌ No flight plan file found in {user_folder}")
        print(f"💡 Expected extensions: {', '.join(valid_extensions)}")
        return

    # Use the first valid file
    user_file = matching_files[0]

    try:
        print(f"📍 Processing: {user_file.name}")

        # Extract and display coordinates briefly
        print_gps_coordinates_table(str(user_file))

        # Generate the map and CSV
        output_path = generate_flight_context_map(
            str(user_file),
            show_coordinates=False,  # No labels on map
            export_coordinates=True  # Generate CSV
        )

        print(f"📁 Files saved to: {OUTPUT_FOLDER}")
        print(f"   • Map: {Path(output_path).name}")
        print(f"   • GPS: {user_file.stem}_gps_coordinates.csv")

    except Exception as e:
        print(f"❌ Error processing {user_file}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Simplified main function - generates only CSV and PNG"""
    
    print("🚁 Drone Flight Map Generator")
    print("=" * 40)
    print("📍 Generates map (PNG) and GPS coordinates (CSV)")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print()
    
    if not QGIS_AVAILABLE:
        print("❌ PyQGIS not available!")
        print("💡 Run with: C:\\OSGeo4W\\bin\\python-qgis.bat drone_map_generator.py")
        return
    
    # Generate map and CSV
    test_with_user_data()

if __name__ == "__main__":
    main()
