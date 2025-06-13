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

# QGIS imports - requires QGIS installation with PyQGIS
try:
    from qgis.core import (
        QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
        QgsPointXY, QgsRectangle, QgsCoordinateReferenceSystem,
        QgsMapSettings, QgsMapRendererCustomPainterJob, QgsMapRendererSequentialJob,
        QgsFeature, QgsGeometry, QgsSymbol, QgsRendererRange, QgsGraduatedSymbolRenderer,
        QgsMarkerSymbol, QgsSimpleMarkerSymbolLayer, QgsLineSymbol, QgsField,
        QgsFields, QgsWkbTypes, QgsFillSymbol, QgsProject, QgsLayerTreeLayer,
        QgsApplication, QgsProcessingFeedback, QgsVectorFileWriter
    )
    from qgis.PyQt.QtCore import QSize, QSizeF
    from qgis.PyQt.QtGui import QColor, QPainter, QImage
    from qgis.PyQt.QtCore import Qt
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    print("⚠️  PyQGIS not available. Please install QGIS with Python bindings.")

# =============================================================================
# 🔧 CONFIGURATION - CHANGE THESE FOR YOUR NEEDS
# =============================================================================

# Output settings
OUTPUT_FOLDER = "flight_maps_output"  # Where to save generated maps
MAP_WIDTH = 2048   # Output image width in pixels
MAP_HEIGHT = 1536  # Output image height in pixels
MAP_DPI = 300      # High DPI for quality

# Context settings - how much area around flight path to show
CONTEXT_BUFFER_PERCENT = 0.3  # Add 30% buffer around flight area (0.5 = 50% more area)
MIN_MAP_SIZE_METERS = 1000    # Minimum map dimension in meters (for very small flight areas)

# Map layer settings
DEFAULT_BASE_LAYER = "Googel Satellite"  # Options: "OpenStreetMap", "Google Satellite", "Bing Aerial"
INCLUDE_TERRAIN = True       # Add terrain/elevation layer
INCLUDE_LABELS = True        # Include place name labels

# Flight path visualization
FLIGHT_PATH_COLOR = "red"    # Color for flight path line
FLIGHT_PATH_WIDTH = 3        # Line width for flight path
WAYPOINT_COLOR = "blue"      # Color for waypoint markers
WAYPOINT_SIZE = 8            # Size of waypoint markers

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

class FlightPlan:
    """Represents a complete flight plan with waypoints and metadata"""
    def __init__(self, points: List[FlightPoint], name: str = "Flight Plan", 
                 description: str = ""):
        self.points = points
        self.name = name
        self.description = description
        self.bounds = self._calculate_bounds()
    
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

# =============================================================================
# FLIGHT PLAN PARSERS
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
            
            points.append(FlightPoint(lat=lat, lon=lon, name=name))
        
        # Parse track points if no waypoints found
        if not points:
            for trkpt in root.findall('.//trkpt' if not namespace else './/gpx:trkpt', namespace):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                points.append(FlightPoint(lat=lat, lon=lon))
        
        return FlightPlan(points=points, name=file_path.stem)
    
    @staticmethod
    def parse_kml(file_path: Path) -> FlightPlan:
        """Parse KML file format"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle KML namespace
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        if not root.tag.endswith('kml'):
            namespace = {}
        
        points = []
        
        # Find coordinates in LineString or Point elements
        for coordinates in root.findall('.//coordinates' if not namespace else './/kml:coordinates', namespace):
            coord_text = coordinates.text.strip()
            for coord_line in coord_text.split('\n'):
                coord_line = coord_line.strip()
                if coord_line:
                    coords = coord_line.split(',')
                    if len(coords) >= 2:
                        lon, lat = float(coords[0]), float(coords[1])
                        alt = float(coords[2]) if len(coords) > 2 else None
                        points.append(FlightPoint(lat=lat, lon=lon, alt=alt))
        
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
# QGIS MAP GENERATOR
# =============================================================================

class QGISMapGenerator:
    """Generate contextual maps using QGIS"""
    
    def __init__(self):
        if not QGIS_AVAILABLE:
            raise ImportError("PyQGIS not available. Please install QGIS with Python bindings.")
        
        # Initialize QGIS Application
        QgsApplication.setPrefixPath("/usr", True)  # Adjust path as needed
        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()
        
        # Create project
        self.project = QgsProject.instance()
        self.project.clear()
        
        logger.info("✅ QGIS initialized successfully")
    
    def generate_contextual_map(self, flight_plan: FlightPlan, 
                              output_filename: str = None) -> str:
        """Generate a contextual map for the flight plan"""
        
        if not flight_plan.points:
            raise ValueError("Flight plan has no points")
        
        logger.info(f"🗺️  Generating contextual map for flight plan: {flight_plan.name}")
        logger.info(f"📍 Flight area: {len(flight_plan.points)} waypoints")
        
        # Calculate map extent with context buffer
        map_extent = self._calculate_map_extent(flight_plan)
        logger.info(f"🔍 Map extent: {map_extent}")
        
        # Setup base layers
        self._add_base_layers()
        
        # Add flight plan layers
        self._add_flight_plan_layers(flight_plan)
        
        # Generate and save map
        if output_filename is None:
            output_filename = f"{flight_plan.name}_contextual_map.png"
        
        output_path = Path(OUTPUT_FOLDER) / output_filename
        self._render_map(map_extent, output_path)
        
        logger.info(f"✅ Map saved: {output_path}")
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
    
    def _add_base_layers(self):
        """Add base map layers"""
        
        # Add OpenStreetMap layer
        if DEFAULT_BASE_LAYER == "OpenStreetMap":
            osm_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            osm_layer = QgsRasterLayer(osm_url, "OpenStreetMap", "wms")
            if osm_layer.isValid():
                self.project.addMapLayer(osm_layer)
                logger.info("✅ Added OpenStreetMap base layer")
            else:
                logger.warning("❌ Failed to add OpenStreetMap layer")
        
        # Add satellite imagery if requested
        elif DEFAULT_BASE_LAYER == "Google Satellite":
            # Note: Check Google's terms of service for your use case
            sat_url = "type=xyz&url=https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
            sat_layer = QgsRasterLayer(sat_url, "Google Satellite", "wms")
            if sat_layer.isValid():
                self.project.addMapLayer(sat_layer)
                logger.info("✅ Added Google Satellite base layer")
            else:
                logger.warning("❌ Failed to add satellite layer")
    
    def _add_flight_plan_layers(self, flight_plan: FlightPlan):
        """Add flight plan waypoints and path as map layers"""
        
        # Create waypoints layer
        waypoints_layer = QgsVectorLayer("Point?crs=EPSG:4326", "Waypoints", "memory")
        waypoints_provider = waypoints_layer.dataProvider()
        
        # Add waypoint features
        features = []
        for i, point in enumerate(flight_plan.points):
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point.lon, point.lat)))
            feature.setAttributes([i, point.name or f"WP{i+1}"])
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
    
    def cleanup(self):
        """Clean up QGIS resources"""
        self.qgs.exitQgis()

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def generate_flight_context_map(flight_plan_file: str, 
                               output_filename: str = None) -> str:
    """
    Main function to generate contextual map from flight plan file
    
    Args:
        flight_plan_file: Path to flight plan file (GPX, KML, CSV, JSON, TXT)
        output_filename: Optional custom output filename
    
    Returns:
        Path to generated map image
    """
    
    logger.info("🚁 Starting Flight Context Map Generation")
    logger.info("=" * 60)
    
    try:
        # Parse flight plan
        flight_plan = FlightPlanParser.parse_file(flight_plan_file)
        logger.info(f"✅ Parsed flight plan: {len(flight_plan.points)} waypoints")
        
        # Generate map
        map_generator = QGISMapGenerator()
        
        try:
            output_path = map_generator.generate_contextual_map(
                flight_plan, output_filename
            )
            
            logger.info("🎉 SUCCESS! Contextual map generated")
            logger.info(f"📁 Output: {output_path}")
            
            return output_path
            
        finally:
            map_generator.cleanup()
    
    except Exception as e:
        logger.error(f"❌ Error generating map: {e}")
        raise

def batch_generate_maps(flight_plans_folder: str) -> List[str]:
    """
    Generate contextual maps for all flight plans in a folder
    
    Args:
        flight_plans_folder: Path to folder containing flight plan files
    
    Returns:
        List of paths to generated map images
    """
    
    folder_path = Path(flight_plans_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {flight_plans_folder}")
    
    generated_maps = []
    
    # Find all supported flight plan files
    flight_files = ["C:/Users/lukev/Downloads/sample_drone_path.kml"]
    for ext in SUPPORTED_FORMATS:
        flight_files.extend(folder_path.glob(f"*{ext}"))
    
    logger.info(f"🔍 Found {len(flight_files)} flight plan files")
    
    for flight_file in flight_files:
        try:
            output_path = generate_flight_context_map(str(flight_file))
            generated_maps.append(output_path)
        except Exception as e:
            logger.error(f"❌ Failed to process {flight_file}: {e}")
    
    logger.info(f"✅ Generated {len(generated_maps)} contextual maps")
    return generated_maps

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def create_sample_flight_plan():
    """Create a sample flight plan for testing"""
    
    # Sample coordinates around a specific area (adjust for your test location)
    sample_points = [
        FlightPoint(lat=40.7128, lon=-74.0060, name="Start Point"),      # NYC area
        FlightPoint(lat=40.7148, lon=-74.0080, name="Waypoint 1"),
        FlightPoint(lat=40.7168, lon=-74.0040, name="Waypoint 2"),
        FlightPoint(lat=40.7138, lon=-74.0020, name="End Point"),
    ]
    
    return FlightPlan(points=sample_points, name="Sample Flight Plan")

def main():
    """Main function for testing and demonstration"""
    
    print("🚁 QGIS Flight Context Map Generator")
    print("=" * 50)
    print("📍 Generates contextual maps showing flight areas with surrounding geography")
    print(f"🗺️  Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print()
    
    if not QGIS_AVAILABLE:
        print("❌ PyQGIS not available!")
        print("💡 Please install QGIS with Python bindings to use this tool")
        return
    
    # Example usage:
    print("💡 Example usage:")
    print("   python drone_map_generator.py")
    print("   # Then call: generate_flight_context_map('your_flight_plan.gpx')")
    print()
    

if __name__ == "__main__":
    main()