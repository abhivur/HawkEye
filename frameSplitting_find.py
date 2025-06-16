#!/usr/bin/env python3
r"""
Simplified Intelligent Drone Mission Planning System
Receives frame data and object detection results as input for smart flight planning.

Run with: python simplified_drone_mission.py
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans

# Import map extraction components
from map_extract import (
    FlightPlan, FlightPoint, FlightPlanParser, 
    QGISMapGenerator, generate_flight_context_map
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MissionType(Enum):
    SEARCH = "search"
    SURVEY = "survey"
    INSPECTION = "inspection"
    MONITORING = "monitoring"

@dataclass
class MissionObjective:
    """Simple mission objective structure"""
    mission_type: MissionType
    target_objects: List[str]
    user_goal: str

@dataclass
class DetectedObject:
    """Object detection result"""
    label: str
    confidence: float
    bbox: Dict[str, float]  # xmin, ymin, xmax, ymax
    area: float

@dataclass
class FrameData:
    """Input frame data with detection results"""
    frame_id: int
    timestamp: float
    frame_path: str  # Path to the frame image
    detected_objects: List[DetectedObject]
    gps_coords: Tuple[float, float, float]  # lat, lon, alt

@dataclass
class SceneObservation:
    """Processed observation from frame data"""
    frame_id: int
    timestamp: float
    gps_coords: Tuple[float, float, float]  # lat, lon, alt
    detected_objects: List[str]
    scene_description: str
    relevance_score: float  # How relevant to mission objective

@dataclass
class OptimizationSuggestion:
    """Suggestion for flight plan optimization"""
    suggestion_type: str  # "route", "capture_rate", "altitude", etc.
    location: Optional[Tuple[float, float]]  # GPS coords if location-specific
    description: str
    reason: str

# =============================================================================
# MISSION OBJECTIVE PARSER
# =============================================================================

class SimpleMissionParser:
    """Parse user objectives into structured format"""
    
    def __init__(self):
        self.target_keywords = {
            "penguin": ["penguin", "penguins", "bird", "wildlife"],
            "ice": ["ice", "glacier", "frozen", "iceberg"],
            "water": ["water", "ocean", "sea", "lake", "river"],
            "vehicle": ["car", "truck", "vehicle", "boat"],
            "building": ["building", "structure", "house", "facility"],
            "person": ["person", "people", "human", "hiker"]
        }
    
    def parse_objective(self, user_input: str) -> MissionObjective:
        """Parse natural language objective"""
        user_input_lower = user_input.lower()
        
        # Determine mission type
        if "find" in user_input_lower or "search" in user_input_lower:
            mission_type = MissionType.SEARCH
        elif "survey" in user_input_lower or "map" in user_input_lower:
            mission_type = MissionType.SURVEY
        elif "inspect" in user_input_lower or "check" in user_input_lower:
            mission_type = MissionType.INSPECTION
        else:
            mission_type = MissionType.MONITORING
        
        # Extract target objects
        target_objects = []
        for target, keywords in self.target_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                target_objects.append(target)
        
        return MissionObjective(
            mission_type=mission_type,
            target_objects=target_objects,
            user_goal=user_input
        )

# =============================================================================
# SCENE CLUSTERING
# =============================================================================

class SceneClusterer:
    """Cluster similar scenes based on detection patterns"""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.model = None
    
    def create_scene_vectors(self, observations: List[SceneObservation]) -> np.ndarray:
        """
        Convert observations to numerical vectors for clustering
        
        Args:
            observations: List of scene observations
            
        Returns:
            NxD numpy array where N is number of observations
        """
        if not observations:
            return np.array([])
        
        # Create a vocabulary of all detected objects
        all_objects = set()
        for obs in observations:
            all_objects.update(obs.detected_objects)
        
        object_vocab = sorted(list(all_objects))
        vocab_size = len(object_vocab)
        
        if vocab_size == 0:
            return np.array([])
        
        # Create binary vectors for each observation
        vectors = []
        for obs in observations:
            vector = np.zeros(vocab_size)
            for obj in obs.detected_objects:
                if obj in object_vocab:
                    idx = object_vocab.index(obj)
                    vector[idx] = 1.0
            
            # Add relevance score as additional feature
            vector = np.append(vector, obs.relevance_score)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def cluster_observations(self, observations: List[SceneObservation]) -> Tuple[np.ndarray, Any]:
        """
        Cluster observations into groups
        
        Args:
            observations: List of scene observations
            
        Returns:
            labels: cluster labels for each observation
            model: trained clustering model
        """
        vectors = self.create_scene_vectors(observations)
        
        if len(vectors) == 0:
            return np.array([]), None
        
        # Adjust number of clusters if we have fewer observations
        n_clusters = min(self.n_clusters, len(observations))
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.model.fit_predict(vectors)
        return labels, self.model
    
    def find_redundant_scenes(self, labels: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Find clusters that appear too frequently (redundant scenes)"""
        if len(labels) == 0:
            return []
        
        unique, counts = np.unique(labels, return_counts=True)
        total_observations = len(labels)
        
        redundant_clusters = []
        for cluster_id, count in zip(unique, counts):
            if count / total_observations > threshold:
                redundant_clusters.append(int(cluster_id))
        
        return redundant_clusters

# =============================================================================
# SCENE ANALYZER (PROCESSES INPUT DATA)
# =============================================================================

class SceneAnalyzer:
    """Analyze scenes from input frame data and detection results"""
    
    def __init__(self):
        self.flight_plan = None
    
    def process_frame_data(self, frame_data_list: List[FrameData], 
                          mission_objective: MissionObjective) -> List[SceneObservation]:
        """
        Process input frame data into scene observations
        
        Args:
            frame_data_list: List of frame data with detection results
            mission_objective: Mission objective with target objects
            
        Returns:
            List of scene observations
        """
        observations = []
        
        for frame_data in frame_data_list:
            if not frame_data.detected_objects:
                continue
            
            # Extract object labels
            object_labels = [obj.label for obj in frame_data.detected_objects]
            
            # Generate scene description
            scene_description = self.generate_scene_description(frame_data.detected_objects, frame_data.frame_id)
            
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(object_labels, mission_objective)
            
            # Create observation
            observation = SceneObservation(
                frame_id=frame_data.frame_id,
                timestamp=frame_data.timestamp,
                gps_coords=frame_data.gps_coords,
                detected_objects=list(set(object_labels)),  # Remove duplicates
                scene_description=scene_description,
                relevance_score=relevance_score
            )
            
            observations.append(observation)
        
        logger.info(f"✅ Processed {len(observations)} scene observations from input data")
        return observations
    
    def generate_scene_description(self, detected_objects: List[DetectedObject], frame_id: int) -> str:
        """Generate natural language description of the scene"""
        if not detected_objects:
            return "No objects detected in this frame"
        
        # Count object types
        object_counts = {}
        for obj in detected_objects:
            label = obj.label
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
        
        # Build description
        descriptions = []
        for label, count in object_counts.items():
            if count == 1:
                descriptions.append(f"a {label}")
            else:
                descriptions.append(f"{count} {label}s")
        
        # Create natural sentence
        if len(descriptions) == 1:
            scene_desc = f"Frame {frame_id} shows {descriptions[0]}"
        elif len(descriptions) == 2:
            scene_desc = f"Frame {frame_id} shows {descriptions[0]} and {descriptions[1]}"
        else:
            scene_desc = f"Frame {frame_id} shows {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        
        return scene_desc
    
    def calculate_relevance_score(self, detected_objects: List[str], mission_objective: MissionObjective) -> float:
        """Calculate how relevant detected objects are to mission objective"""
        if not detected_objects or not mission_objective.target_objects:
            return 0.0
        
        # Simple keyword matching for relevance
        matches = 0
        total_targets = len(mission_objective.target_objects)
        
        for target in mission_objective.target_objects:
            target_lower = target.lower()
            for detected in detected_objects:
                detected_lower = detected.lower()
                if target_lower in detected_lower or detected_lower in target_lower:
                    matches += 1
                    break  # Don't double count
        
        return min(matches / total_targets, 1.0)

# =============================================================================
# FLIGHT PLAN OPTIMIZER
# =============================================================================

class FlightPlanOptimizer:
    """Optimize flight plans based on observations and objectives"""
    
    def __init__(self):
        self.scene_clusterer = SceneClusterer()
    
    def save_flight_plan_as_kml(self, flight_plan: FlightPlan, output_path: str):
        """
        Save flight plan as KML file
        
        Args:
            flight_plan: FlightPlan object to save
            output_path: Output file path for KML
        """
        kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{flight_plan.name}</name>
    <description>{flight_plan.description}</description>
    <Style id="lineStyle">
      <LineStyle>
        <color>ff0000ff</color>
        <width>4</width>
      </LineStyle>
    </Style>
    <Style id="waypointStyle">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <Folder>
      <name>Waypoints</name>
'''
        
        # Add waypoints as placemarks
        for i, point in enumerate(flight_plan.points):
            altitude = point.alt if point.alt else 100
            kml_content += f'''      <Placemark>
        <name>{point.name if point.name else f"WP{i+1}"}</name>
        <styleUrl>#waypointStyle</styleUrl>
        <Point>
          <coordinates>{point.lon},{point.lat},{altitude}</coordinates>
          <altitudeMode>relativeToGround</altitudeMode>
        </Point>
      </Placemark>
'''
        
        kml_content += '''    </Folder>
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#lineStyle</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>relativeToGround</altitudeMode>
        <coordinates>
'''
        
        # Add path coordinates
        for point in flight_plan.points:
            altitude = point.alt if point.alt else 100
            kml_content += f'          {point.lon},{point.lat},{altitude}\n'
        
        kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(kml_content)
        
        logger.info(f"✅ Saved flight plan to: {output_path}")
    
    def optimize_flight_plan(self, 
                           original_plan: FlightPlan,
                           observations: List[SceneObservation],
                           mission_objective: MissionObjective) -> Tuple[FlightPlan, List[OptimizationSuggestion]]:
        """
        Optimize flight plan based on observations and mission objective
        
        Returns:
            optimized_plan: New optimized flight plan
            suggestions: List of optimization suggestions
        """
        suggestions = []
        optimized_points = original_plan.points.copy()
        
        # Analyze observations for mission-relevant areas
        high_value_areas = []
        low_value_areas = []
        obstacles = []
        
        for obs in observations:
            # Check if observation is relevant to mission
            if any(target in obs.detected_objects for target in mission_objective.target_objects):
                high_value_areas.append(obs)
            elif obs.relevance_score < 0.3:
                low_value_areas.append(obs)
            
            # Check for obstacles
            if "wall" in obs.scene_description.lower() or "blocked" in obs.scene_description.lower():
                obstacles.append(obs)
        
        # Generate optimization suggestions
        if high_value_areas:
            # Suggest focusing on high-value areas
            for area in high_value_areas:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="route",
                    location=(area.gps_coords[0], area.gps_coords[1]),
                    description=f"Add more waypoints around GPS {area.gps_coords[0]:.4f}, {area.gps_coords[1]:.4f}",
                    reason=f"High concentration of {mission_objective.target_objects} detected"
                ))
        
        if obstacles:
            # Suggest avoiding obstacles
            for obstacle in obstacles:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="route",
                    location=(obstacle.gps_coords[0], obstacle.gps_coords[1]),
                    description=f"Avoid area around GPS {obstacle.gps_coords[0]:.4f}, {obstacle.gps_coords[1]:.4f}",
                    reason=obstacle.scene_description
                ))
        
        # Optimize capture rate based on scene diversity
        self._suggest_capture_rate_optimization(observations, suggestions)
        
        # Create optimized flight plan
        optimized_plan = self._create_optimized_plan(
            original_plan, high_value_areas, obstacles
        )
        
        return optimized_plan, suggestions
    
    def _suggest_capture_rate_optimization(self, 
                                         observations: List[SceneObservation],
                                         suggestions: List[OptimizationSuggestion]):
        """Suggest capture rate changes based on scene redundancy"""
        # Group observations by general area
        area_groups = {}
        for obs in observations:
            area_key = (round(obs.gps_coords[0], 3), round(obs.gps_coords[1], 3))
            if area_key not in area_groups:
                area_groups[area_key] = []
            area_groups[area_key].append(obs)
        
        # Check for redundant areas
        for area, obs_list in area_groups.items():
            if len(obs_list) > 10:  # Many observations in same area
                # Check if scenes are similar
                unique_objects = set()
                for obs in obs_list:
                    unique_objects.update(obs.detected_objects)
                
                if len(unique_objects) < 3:  # Low diversity
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="capture_rate",
                        location=area,
                        description=f"Reduce capture rate to 5 fps around GPS {area[0]:.4f}, {area[1]:.4f}",
                        reason="Low scene diversity detected - mostly redundant frames"
                    ))
    
    def _create_optimized_plan(self,
                             original_plan: FlightPlan,
                             high_value_areas: List[SceneObservation],
                             obstacles: List[SceneObservation]) -> FlightPlan:
        """Create optimized flight plan"""
        optimized_points = []
        
        # Start with original points
        for point in original_plan.points:
            # Check if point is near an obstacle
            near_obstacle = False
            for obstacle in obstacles:
                distance = self._calculate_distance(
                    point.lat, point.lon,
                    obstacle.gps_coords[0], obstacle.gps_coords[1]
                )
                if distance < 0.001:  # ~100 meters
                    near_obstacle = True
                    break
            
            if not near_obstacle:
                optimized_points.append(point)
        
        # Add extra waypoints near high-value areas
        for area in high_value_areas[:3]:  # Limit to top 3 areas
            # Add circular pattern around high-value area
            center_lat, center_lon = area.gps_coords[0], area.gps_coords[1]
            for angle in [0, 90, 180, 270]:
                offset = 0.0005  # ~50 meters
                lat_offset = offset * np.cos(np.radians(angle))
                lon_offset = offset * np.sin(np.radians(angle))
                
                new_point = FlightPoint(
                    lat=center_lat + lat_offset,
                    lon=center_lon + lon_offset,
                    alt=100,
                    name=f"search_pattern_{angle}"
                )
                optimized_points.append(new_point)
        
        return FlightPlan(
            points=optimized_points,
            name=f"{original_plan.name}_optimized",
            description=f"Optimized for: {high_value_areas[0].detected_objects if high_value_areas else 'general'}"
        )
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Simple distance calculation (in degrees)"""
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

# =============================================================================
# MISSION COORDINATOR
# =============================================================================

class SimpleMissionCoordinator:
    """Coordinate the mission planning process"""
    
    def __init__(self):
        self.parser = SimpleMissionParser()
        self.scene_analyzer = SceneAnalyzer()
        self.optimizer = FlightPlanOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def plan_mission_from_data(self, 
                              flight_plan_file: str,
                              user_objective: str,
                              frame_data_list: List[FrameData]) -> Dict[str, Any]:
        """
        Main mission planning function using input frame data
        
        Args:
            flight_plan_file: Path to flight plan file
            user_objective: User's mission objective
            frame_data_list: List of frame data with detection results
            
        Returns:
            Comprehensive mission plan with observations and suggestions
        """
        self.logger.info(f"🚁 Starting mission planning from input data")
        self.logger.info(f"📋 Objective: {user_objective}")
        
        # Parse objective
        mission_objective = self.parser.parse_objective(user_objective)
        self.logger.info(f"🎯 Mission type: {mission_objective.mission_type.value}")
        self.logger.info(f"🔍 Target objects: {mission_objective.target_objects}")
        
        # Load flight plan
        original_plan = FlightPlanParser.parse_file(flight_plan_file)
        self.logger.info(f"📍 Loaded flight plan with {len(original_plan.points)} waypoints")
        
        # Process frame data into observations
        self.logger.info(f"🎥 Processing {len(frame_data_list)} frames of input data...")
        observations = self.scene_analyzer.process_frame_data(frame_data_list, mission_objective)
        
        # Perform scene clustering analysis
        if observations:
            labels, _ = self.optimizer.scene_clusterer.cluster_observations(observations)
            redundant_scenes = self.optimizer.scene_clusterer.find_redundant_scenes(labels)
            
            self.logger.info(f"📊 Found {len(observations)} scene observations")
            self.logger.info(f"🔄 Identified {len(redundant_scenes)} redundant scene types")
        
        # Optimize flight plan
        optimized_plan, suggestions = self.optimizer.optimize_flight_plan(
            original_plan, observations, mission_objective
        )
        
        # Generate report
        report = self._generate_report(
            mission_objective, original_plan, optimized_plan,
            observations, suggestions
        )
        
        # Save optimized flight plan as KML
        optimized_kml_path = f"optimized_{Path(flight_plan_file).stem}.kml"
        self.optimizer.save_flight_plan_as_kml(optimized_plan, optimized_kml_path)
        report['optimized_flight_plan_file'] = optimized_kml_path
        
        self.logger.info("✅ Mission planning complete!")
        
        return report
    
    def _generate_report(self,
                        mission_objective: MissionObjective,
                        original_plan: FlightPlan,
                        optimized_plan: FlightPlan,
                        observations: List[SceneObservation],
                        suggestions: List[OptimizationSuggestion]) -> Dict[str, Any]:
        """Generate comprehensive mission report"""
        
        # Format observations for output
        formatted_observations = []
        for obs in observations:
            formatted_observations.append({
                "frame_id": obs.frame_id,
                "timestamp": obs.timestamp,
                "gps_location": {
                    "latitude": obs.gps_coords[0],
                    "longitude": obs.gps_coords[1],
                    "altitude": obs.gps_coords[2]
                },
                "detected_objects": obs.detected_objects,
                "description": obs.scene_description,
                "relevance_to_mission": obs.relevance_score
            })
        
        # Format suggestions
        formatted_suggestions = []
        for sugg in suggestions:
            formatted_suggestions.append({
                "type": sugg.suggestion_type,
                "location": {
                    "latitude": sugg.location[0] if sugg.location else None,
                    "longitude": sugg.location[1] if sugg.location else None
                },
                "suggestion": sugg.description,
                "reasoning": sugg.reason
            })
        
        # Key findings summary
        key_findings = []
        
        # Find target objects
        target_found_locations = []
        for obs in observations:
            if any(target in obs.detected_objects for target in mission_objective.target_objects):
                target_found_locations.append(obs)
                key_findings.append(
                    f"✅ {', '.join([t for t in mission_objective.target_objects if t in obs.detected_objects])} "
                    f"found at GPS {obs.gps_coords[0]:.6f}, {obs.gps_coords[1]:.6f}"
                )
        
        # Environmental observations
        env_features = {"water": [], "ice": [], "obstacles": []}
        for obs in observations:
            if "water" in obs.scene_description.lower():
                env_features["water"].append(obs)
            if "ice" in obs.scene_description.lower() or "ice" in obs.detected_objects:
                env_features["ice"].append(obs)
            if "wall" in obs.scene_description.lower() or "blocked" in obs.scene_description.lower():
                env_features["obstacles"].append(obs)
        
        if env_features["water"]:
            key_findings.append(f"💧 Water detected in {len(env_features['water'])} locations")
        if env_features["ice"]:
            key_findings.append(f"🧊 Ice formations in {len(env_features['ice'])} areas")
        if env_features["obstacles"]:
            key_findings.append(f"⚠️ {len(env_features['obstacles'])} obstacles detected")
        
        return {
            "mission_objective": {
                "user_goal": mission_objective.user_goal,
                "mission_type": mission_objective.mission_type.value,
                "target_objects": mission_objective.target_objects
            },
            "flight_plans": {
                "original_waypoints": len(original_plan.points),
                "optimized_waypoints": len(optimized_plan.points),
                "optimization_applied": len(optimized_plan.points) != len(original_plan.points)
            },
            "key_findings": key_findings,
            "scene_observations": formatted_observations,
            "optimization_suggestions": formatted_suggestions,
            "summary": {
                "total_observations": len(observations),
                "high_relevance_scenes": len([o for o in observations if o.relevance_score > 0.7]),
                "target_objects_found": len(target_found_locations) > 0,
                "suggested_optimizations": len(suggestions)
            }
        }

# =============================================================================
# UTILITY FUNCTIONS FOR DATA INPUT
# =============================================================================

def create_frame_data_from_json(json_file_path: str) -> List[FrameData]:
    """
    Load frame data from a JSON file
    
    Expected JSON format:
    [
        {
            "frame_id": 1,
            "timestamp": 1.0,
            "frame_path": "/path/to/frame.jpg",
            "gps_coords": [lat, lon, alt],
            "detected_objects": [
                {
                    "label": "person",
                    "confidence": 0.85,
                    "bbox": {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200},
                    "area": 16200
                }
            ]
        }
    ]
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    frame_data_list = []
    for frame_dict in data:
        # Convert detected objects
        detected_objects = []
        for obj_dict in frame_dict.get('detected_objects', []):
            detected_obj = DetectedObject(
                label=obj_dict['label'],
                confidence=obj_dict['confidence'],
                bbox=obj_dict['bbox'],
                area=obj_dict['area']
            )
            detected_objects.append(detected_obj)
        
        # Create frame data
        frame_data = FrameData(
            frame_id=frame_dict['frame_id'],
            timestamp=frame_dict['timestamp'],
            frame_path=frame_dict['frame_path'],
            detected_objects=detected_objects,
            gps_coords=tuple(frame_dict['gps_coords'])
        )
        
        frame_data_list.append(frame_data)
    
    return frame_data_list

def create_sample_frame_data() -> List[FrameData]:
    """Create sample frame data for testing"""
    sample_data = [
        FrameData(
            frame_id=1,
            timestamp=1.0,
            frame_path="frame_001.jpg",
            detected_objects=[
                DetectedObject(label="person", confidence=0.85, bbox={"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}, area=16200)
            ],
            gps_coords=(40.7128, -74.0060, 100)
        ),
        FrameData(
            frame_id=2,
            timestamp=2.0,
            frame_path="frame_002.jpg",
            detected_objects=[
                DetectedObject(label="penguin", confidence=0.92, bbox={"xmin": 50, "ymin": 60, "xmax": 150, "ymax": 250}, area=19000),
                DetectedObject(label="ice", confidence=0.78, bbox={"xmin": 200, "ymin": 100, "xmax": 400, "ymax": 300}, area=40000)
            ],
            gps_coords=(40.7130, -74.0058, 100)
        )
    ]
    return sample_data

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    """Main execution function"""
    print("🚁 Simplified Intelligent Drone Mission System (Input-Based)")
    print("=" * 60)
    
    # Get user inputs
    flight_plan_file = input("📁 Flight plan file (KML/GPX/CSV): ").strip()
    if not Path(flight_plan_file).exists():
        print(f"❌ File not found: {flight_plan_file}")
        return
    
    print("\n💡 Example objectives:")
    print('  - "Find penguins in this area"')
    print('  - "Survey the coastline for ice formations"')
    print('  - "Should I capture at 10fps everywhere?"')
    
    user_objective = input("\n🎯 Your mission objective: ").strip()
    
    # Get frame data source
    print("\n📊 Frame data input options:")
    print("  1. Load from JSON file")
    print("  2. Use sample data for testing")
    
    data_choice = input("Choose option (1 or 2): ").strip()
    
    frame_data_list = []
    
    if data_choice == "1":
        json_file = input("📁 JSON file with frame data: ").strip()
        if Path(json_file).exists():
            try:
                frame_data_list = create_frame_data_from_json(json_file)
                print(f"✅ Loaded {len(frame_data_list)} frames from JSON")
            except Exception as e:
                print(f"❌ Error loading JSON: {e}")
                return
        else:
            print(f"❌ File not found: {json_file}")
            return
    elif data_choice == "2":
        frame_data_list = create_sample_frame_data()
        print(f"✅ Using {len(frame_data_list)} sample frames")
    else:
        print("❌ Invalid choice")
        return
    
    if not frame_data_list:
        print("❌ No frame data available")
        return
    
    # Run mission planning
    coordinator = SimpleMissionCoordinator()
    
    try:
        report = coordinator.plan_mission_from_data(
            flight_plan_file=flight_plan_file,
            user_objective=user_objective,
            frame_data_list=frame_data_list
        )
        
        # Display results
        print("\n" + "="*60)
        print("📋 MISSION PLANNING REPORT")
        print("="*60)
        
        print(f"\n🎯 Mission: {report['mission_objective']['user_goal']}")
        print(f"Type: {report['mission_objective']['mission_type']}")
        print(f"Targets: {', '.join(report['mission_objective']['target_objects'])}")
        
        print("\n🔑 KEY FINDINGS:")
        for finding in report['key_findings']:
            print(f"  {finding}")
        
        print("\n📊 SUMMARY:")
        summary = report['summary']
        print(f"  Total observations: {summary['total_observations']}")
        print(f"  High relevance scenes: {summary['high_relevance_scenes']}")
        print(f"  Target found: {'Yes' if summary['target_objects_found'] else 'No'}")
        print(f"  Optimization suggestions: {summary['suggested_optimizations']}")
        
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for i, sugg in enumerate(report['optimization_suggestions'][:5], 1):
            print(f"\n  {i}. {sugg['suggestion']}")
            print(f"     Reason: {sugg['reasoning']}")
            if sugg['location']['latitude']:
                print(f"     Location: {sugg['location']['latitude']:.6f}, {sugg['location']['longitude']:.6f}")
        
        print("\n📋 DETAILED OBSERVATIONS:")
        for i, obs in enumerate(report['scene_observations'][:10], 1):  # Show first 10
            print(f"\n  Frame {obs['frame_id']}:")
            print(f"    Objects: {', '.join(obs['detected_objects'])}")
            print(f"    Description: {obs['description']}")
            print(f"    Relevance: {obs['relevance_to_mission']:.2f}")
            print(f"    GPS: {obs['gps_location']['latitude']:.6f}, {obs['gps_location']['longitude']:.6f}")
        
        if len(report['scene_observations']) > 10:
            print(f"\n  ... and {len(report['scene_observations']) - 10} more observations")
        
        # Save detailed report
        output_file = f"mission_report_{Path(flight_plan_file).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {output_file}")
        
        # Show optimized flight plan file
        if 'optimized_flight_plan_file' in report:
            print(f"✅ Optimized flight plan saved to: {report['optimized_flight_plan_file']}")
            print(f"   You can load this KML file in any mapping software!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Mission planning failed: {e}", exc_info=True)


def demo_json_format():
    """Print example JSON format for frame data"""
    print("\n📄 EXAMPLE JSON FORMAT FOR FRAME DATA:")
    print("="*50)
    
    example_json = {
        "frame_data": [
            {
                "frame_id": 1,
                "timestamp": 1.0,
                "frame_path": "/path/to/frame_000001.jpg",
                "gps_coords": [40.7128, -74.0060, 100],
                "detected_objects": [
                    {
                        "label": "person",
                        "confidence": 0.85,
                        "bbox": {
                            "xmin": 10,
                            "ymin": 20, 
                            "xmax": 100,
                            "ymax": 200
                        },
                        "area": 16200
                    },
                    {
                        "label": "penguin",
                        "confidence": 0.92,
                        "bbox": {
                            "xmin": 150,
                            "ymin": 300,
                            "xmax": 250,
                            "ymax": 500
                        },
                        "area": 20000
                    }
                ]
            },
            {
                "frame_id": 2,
                "timestamp": 2.0,
                "frame_path": "/path/to/frame_000002.jpg",
                "gps_coords": [40.7130, -74.0058, 100],
                "detected_objects": [
                    {
                        "label": "ice",
                        "confidence": 0.78,
                        "bbox": {
                            "xmin": 0,
                            "ymin": 0,
                            "xmax": 400,
                            "ymax": 300
                        },
                        "area": 120000
                    }
                ]
            }
        ]
    }
    
    print(json.dumps(example_json, indent=2))
    print("\n💡 Save this format as a .json file to use with the system")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-json":
        demo_json_format()
    else:
        main()
