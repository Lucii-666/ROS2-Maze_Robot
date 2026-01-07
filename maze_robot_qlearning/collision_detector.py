#!/usr/bin/env python3
"""
Collision Detector for Maze Navigation
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class CollisionDetector(Node):
    def __init__(self):
        super().__init__('collision_detector')
        
        # Collision detection parameters
        self.collision_threshold = 0.3  # meters
        self.warning_threshold = 0.5  # meters
        
        # Laser scan data
        self.latest_scan = None
        self.collision_detected = False
        self.warning_zone = False
        
        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Collision Detector initialized')
    
    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg
        self._check_collision()
    
    def _check_collision(self):
        """Check if collision is imminent"""
        if self.latest_scan is None:
            return
        
        ranges = np.array(self.latest_scan.ranges)
        
        # Filter out invalid readings
        valid_ranges = ranges[(ranges > self.latest_scan.range_min) & 
                             (ranges < self.latest_scan.range_max)]
        
        if len(valid_ranges) == 0:
            return
        
        min_distance = np.min(valid_ranges)
        
        # Check collision
        if min_distance < self.collision_threshold:
            self.collision_detected = True
            self.warning_zone = False
            self.get_logger().warn(f'COLLISION DETECTED! Distance: {min_distance:.2f}m')
        elif min_distance < self.warning_threshold:
            self.collision_detected = False
            self.warning_zone = True
            self.get_logger().info(f'Warning zone. Distance: {min_distance:.2f}m')
        else:
            self.collision_detected = False
            self.warning_zone = False
    
    def is_collision(self):
        """Check if collision is detected"""
        return self.collision_detected
    
    def is_warning(self):
        """Check if in warning zone"""
        return self.warning_zone
    
    def get_closest_obstacle_distance(self):
        """Get distance to closest obstacle"""
        if self.latest_scan is None:
            return float('inf')
        
        ranges = np.array(self.latest_scan.ranges)
        valid_ranges = ranges[(ranges > self.latest_scan.range_min) & 
                             (ranges < self.latest_scan.range_max)]
        
        if len(valid_ranges) == 0:
            return float('inf')
        
        return np.min(valid_ranges)
    
    def get_obstacle_direction(self):
        """Get direction of closest obstacle"""
        if self.latest_scan is None:
            return None
        
        ranges = np.array(self.latest_scan.ranges)
        min_idx = np.argmin(ranges)
        
        angle = self.latest_scan.angle_min + min_idx * self.latest_scan.angle_increment
        
        return angle


def main(args=None):
    rclpy.init(args=args)
    detector = CollisionDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
