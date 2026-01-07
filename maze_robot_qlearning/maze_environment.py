#!/usr/bin/env python3
"""
Maze Environment Interface for Robot Navigation
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import numpy as np


class MazeEnvironment(Node):
    def __init__(self):
        super().__init__('maze_environment')
        
        # Environment parameters
        self.grid_size = 10
        self.cell_size = 1.0
        self.current_position = None
        self.goal_position = (9, 9)
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.get_logger().info('Maze Environment initialized')
    
    def odom_callback(self, msg):
        """Update current position from odometry"""
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
    
    def get_state(self):
        """Convert continuous position to discrete grid state"""
        if self.current_position is None:
            return (0, 0)
        
        x = int(self.current_position[0] / self.cell_size)
        y = int(self.current_position[1] / self.cell_size)
        return (x, y)
    
    def execute_action(self, action):
        """Execute action: 0=forward, 1=backward, 2=left, 3=right"""
        cmd = Twist()
        
        if action == 0:  # Forward
            cmd.linear.x = 0.5
        elif action == 1:  # Backward
            cmd.linear.x = -0.5
        elif action == 2:  # Left
            cmd.angular.z = 0.5
        elif action == 3:  # Right
            cmd.angular.z = -0.5
        
        self.cmd_vel_pub.publish(cmd)
    
    def is_goal_reached(self):
        """Check if robot reached the goal"""
        state = self.get_state()
        return state == self.goal_position


def main(args=None):
    rclpy.init(args=args)
    environment = MazeEnvironment()
    rclpy.spin(environment)
    environment.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
