import pygame
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import scipy

from utils import scale_image

#################### Global variables ####################
# Images that will be displayed 
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/elipse-track.png"), 2.0)

# Image that represent the borders, that is, the white lines
TRACK_BORDER = scale_image(pygame.image.load("imgs/elipse-border.png"), 2.0)

# Mask of the borders (utilize for the collisions)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

# Image of the car
CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)

# Widht and Height 
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

# Minimum distance between the car and the lines
LTA_MIN_DISTANCE = 7

FPS = 60

#################### Car functions ####################

# Max speed (MAX_V m/s) and angular velocity (MAX_PHI_OMEGA rad/s)
MAX_V = 33.0 # equivalent to 120 km/h
MAX_PHI_OMEGA = 2*np.pi/9 # Maximum steering velocity in rad/s

MAX_PHI = np.pi/9 # Maximum steering angle in rad
LENGTH_CAR_AXLES = 3 # Length between axles in meters
REALIGN_STEERING = 0.01 # Value to realign the steering road to the straight road (This is like the acelaration to realign the steering road to the straight road)

STEERING_SMOOTH = 0.5 # Value to smooth the steering road (i.e to make the car go back to the tracjetory smoothly) Not used

INCREMENT_V = 1 # Increment for velocity (m/s)
INCREMENT_PHI_OMEGA = 0.2 # Increment for steering velocity (rad/s) Not used

# A class where is defined all the functions that have something to do with the car modelation (Kinematics, ...)
class Car:
    '''
    Robot with differential drive dynamics
    x = robot state
        x[0] = position x (m)
        x[1] = position y (m)
        x[2] = heading theta (rad)
        x[3] = steering angle phi (rad)
    u = controls
        u[0] = v, forward velocity (m/s)
        u[1] = phi_omega, angular velocity of the steering (rad/s)
    EOM = equations of motion
        xdot[0] = v*cos(theta)
        xdot[1] = v*sin(theta)
        xdot[2] = 1/L*v*tan(phi)
        xdot[3] = phi_omega
    '''
    max_v = MAX_V
    max_phi_omega = MAX_PHI_OMEGA
    max_phi = MAX_PHI
    L = LENGTH_CAR_AXLES

    realign_steering = REALIGN_STEERING

    steering_smooth = STEERING_SMOOTH

    v_increment = INCREMENT_V
    phi_increment = INCREMENT_PHI_OMEGA
    
    # Init function
    def __init__(self, x_init, environment):
        self.car_image = CAR
        self.set_state(x_init)

        self.ray = Ray(x_init, environment)

    def set_state1(self, x):
        self.x = x
        self.x[2] = np.arctan2(np.sin(x[2]),np.cos(x[2])) # Keep angle between -pi and +pi

    def set_state(self, x):
        '''
        Implementation of the kinematic ackermann model
        x = [x, y, theta, phi] 
        '''
        self.x = x
        self.x[2] = np.arctan2(np.sin(x[2]),np.cos(x[2]))
        self.x[3] = self.get_phi(0)
    
    def get_phi(self, phi_omega):
        '''
        Clamp the steering angle between -max_phi and +max_phi
        Realign the steering road to the straight road if phi_omega = 0
        '''
        self.x[3] = np.arctan2(np.sin(self.x[3]),np.cos(self.x[3]))
        
        if abs(phi_omega) > 0:
            if self.x[3] > 0:
                return min(self.max_phi, self.x[3])
            elif self.x[3] < 0:
                return max(-self.max_phi, self.x[3])
            else:
                return 0
        else: # case where phi_omega = 0 and I want to realign the steering road to the straight road
            if self.x[3] > 0:
                return min(self.max_phi, max(self.x[3] - self.realign_steering, 0))
            elif self.x[3] < 0:
                return max(-self.max_phi, min(self.x[3] + self.realign_steering, 0))
            else:
                return 0
        
    def get_pose(self):
        return self.x
    
    def move_step(self, u, dt, phi, lta_activation, environment):
        if lta_activation == 1 and u[1] == 0:
            self.x[3] += phi * self.steering_smooth

        y = np.zeros(6)
        y[:4] = self.x; y[4:] = u
        result = scipy.integrate.solve_ivp(self.EOM,[0,dt],y)
        self.x = result.y[:4,-1]
        self.x[2] = np.arctan2(np.sin(self.x[2]),np.cos(self.x[2]))
        self.x[3] = self.get_phi(u[1])

        self.pose_estimator

        # Update the Ray
        self.ray.set_state(self.x, environment)

    def EOM(self, t, y):
        px = y[0]; py = y[1]; theta = y[2]; phi = y[3]
        v = max(min(y[4],self.max_v),-self.max_v); phi_omega = max(min(y[5],self.max_phi_omega),-self.max_phi_omega) # forward and angular velocity
        ydot = np.zeros(6)
        ydot[0] = v*np.cos(theta)
        ydot[1] = v*np.sin(theta)
        ydot[2] = 1/self.L * v * np.tan(phi)
        ydot[3] = phi_omega
        ydot[4] = 0
        ydot[5] = 0
        return ydot
    
    def pose_estimator(self, u, dt):
        '''
        Do the estimation of the pose and the sensor simulation
        '''
        # Simulate the sensor measures
        # Gaussian noise
        noise_std = 0.01  # standart deviation (Ajust if necessary)
        x_measure = self.x[0] + np.random.normal(0, noise_std)  
        y_measure = self.x[1] + np.random.normal(0, noise_std)  
        theta_measure = self.x[2] + np.random.normal(0, noise_std / 10)  
        phi_measure = self.x[3] + np.random.normal(0, noise_std / 10)  
        
        # Apply breaking disturb with the probability of 5%
        disturb_probability = 0.05
        if np.random.rand() < disturb_probability:
            disturb_type = np.random.choice(['x', 'y', 'theta', 'phi'])
            
            if disturb_type == 'x':
                x_measure *= 0.5 
                self.x[1:] = y_measure, theta_measure, phi_measure
            elif disturb_type == 'y':
                y_measure *= 0.5  
                self.x[0] = x_measure
                self.x[2:] = theta_measure, phi_measure
            elif disturb_type == 'theta':
                theta_measure += np.random.normal(0, 0.5)  
                self.x[:2] = x_measure, y_measure
                self.x[3] = theta_measure
            elif disturb_type == 'phi':
                phi_measure += np.random.normal(0, 0.5)
                self.x[:3] = x_measure, y_measure, theta_measure

    
    def update_u(self, u, key_event):
        '''
        Update controls with constraints:
        - Increment `u[1]` when left or right is pressed, up to a maximum value.
        - Increment `u[0]` when up is pressed and keep the value constant.
        '''
        if key_event.type == pygame.KEYDOWN:
            if key_event.key == pygame.K_LEFT:
                # Incrementally decrease `u[1]` down to -max_phi_omega
                #u[1] = max(-self.max_phi_omega, u[1] - self.phi_increment)
                u[1] = -self.max_phi_omega
            elif key_event.key == pygame.K_RIGHT:
                # Incrementally increase `u[1]` up to max_phi_omega
                #u[1] = min(self.max_phi_omega, u[1] + self.phi_increment)
                u[1] = self.max_phi_omega
            elif key_event.key == pygame.K_UP:
                # Incrementally increase `u[0]` up to max_v
                u[0] = min(self.max_v, u[0] + self.v_increment)
            elif key_event.key == pygame.K_DOWN:
                # Incrementally decrease `u[0]` up to 0
                u[0] = max(0, u[0] - self.v_increment)

        if key_event.type == pygame.KEYUP:
            if key_event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                # Stop steering when keys are released
                u[1] = 0

            if key_event.key == pygame.K_UP:
                # Keep velocity constant when up is released
                pass  # No change to u[0]
            elif key_event.key == pygame.K_DOWN:
                # Stop the car when down is released
                u[0] = 0

        return u
    
#################### Sensor functions ####################

RAY_MAX_LENGTH = 10 # Ray max length in m
RAY_ANGLE = np.pi/4 # Ray angle of direction in rad 

class Ray:
    def __init__(self, x_init, environment):                
        self.max_length = RAY_MAX_LENGTH
        self.ray_length = RAY_MAX_LENGTH
        self.ray_angle = RAY_ANGLE
        self.sensor_line_right = None # Vextor of points that represent the right sensor line
        self.sensor_line_left = None # Vextor of points that represent the left sensor line 

        self.set_state(x_init, environment)

    def set_state(self, x, environment):
        self.x = x
        self.x[2] = np.arctan2(np.sin(x[2]),np.cos(x[2])) # Keep angle between -pi and +pi

        self.create_ray(environment)
        
    def create_ray(self, environment):
        ray_position = self.x[:2]  # Ray's position that coincides with the car's position

        # Ray's left line
        ray_left_dx = self.max_length * np.cos(self.x[2] - self.ray_angle)
        ray_left_dy = self.max_length * np.sin(self.x[2] - self.ray_angle)
        ray_left_end = (ray_position[0] + ray_left_dx, ray_position[1] + ray_left_dy)

        #Check if the ray collides with the window borders
        ray_left_end = environment.touch_border_window(ray_left_end)
        
        self.sensor_line_left = (ray_position, ray_left_end)
        
        # Ray's right line
        ray_right_dx = self.max_length * np.cos(self.x[2] + self.ray_angle)
        ray_right_dy = self.max_length * np.sin(self.x[2] + self.ray_angle)
        ray_right_end = (ray_position[0] + ray_right_dx, ray_position[1] + ray_right_dy)
        
        #Check if the ray collides with the window borders
        ray_right_end = environment.touch_border_window(ray_right_end)
    
        self.sensor_line_right = (ray_position, ray_right_end)
    
    def collide(self, environment, track_mask):
        collision_point = np.array([[-1.0, -1.0], [-1.0, -1.0]])

        car_position_px = environment.position2pixel(self.sensor_line_right[0]) # This will coincide with the start of the ray 

        # Ray Right line
        # Convert sensor line from meters to pixels
        ray_end_right_px = environment.position2pixel(self.sensor_line_right[1])       
        
        # Detect overlap
        poi_right = self.find_closest_collision_binary(environment, track_mask, car_position_px, ray_end_right_px)
        
        if poi_right is not None:

            # Guassian noise
            noise_std = 0.01  # Standar Deviation 
            poi_right[0] += np.random.normal(0, noise_std)  
            poi_right[1] += np.random.normal(0, noise_std)  
            
            # Apply breaking disturb with probability of 5%
            disturb_probability = 0.05
            if np.random.rand() < disturb_probability:
                poi_right = [-1.0, -1.0]           

            collision_point[0, 0], collision_point[0, 1] = poi_right[0], poi_right[1]

        # Ray Left line
        # Convert sensor line from meters to pixels
        ray_end_left_px = environment.position2pixel(self.sensor_line_left[1])

        # Detect overlap
        poi_left = self.find_closest_collision_binary(environment, track_mask, car_position_px, ray_end_left_px)

        if poi_left is not None:

            # Guassian noise
            noise_std = 0.01  # Standar Deviation 
            poi_left[0] += np.random.normal(0, noise_std)  
            poi_left[1] += np.random.normal(0, noise_std)  
            
            # Apply breaking disturb with probability of 5%
            disturb_probability = 0.05
            if np.random.rand() < disturb_probability:
                poi_left = [-1.0, -1.0]


            collision_point[1, 0], collision_point[1, 1] = poi_left[0], poi_left[1]

        return collision_point
 
    # Auxiliar function to find the closest collision point
    def find_closest_collision_binary(self, environment, track_mask, ray_start_px, ray_end_px):
        start = np.array(ray_start_px)
        end = np.array(ray_end_px)

        closest_point = None

        while np.linalg.norm(end - start) > 1:  # Stop when the segment length is <= 1 pixel
            midpoint = (start + end) / 2
            midpoint_px = (int(midpoint[0]), int(midpoint[1]))

            if 0 <= midpoint_px[0] < track_mask.get_size()[0] and 0 <= midpoint_px[1] < track_mask.get_size()[1]:
                if track_mask.get_at(midpoint_px):  # Collision detected
                    closest_point = environment.pixel2position(midpoint_px)
                    end = midpoint  # Search toward the start of the ray
                else:
                    start = midpoint  # Search toward the end of the ray
        
        return closest_point


#################### Environment functions ####################
class Environment:
    METER_PER_PIXEL = 0.12 # Conversion where the car size has aproximately 5 meters
    #METER_PER_PIXEL = 0.02
    # METER_PER_PIXEL = 0.025 # Conversion from meters to pixels
    # METER_PER_PIXEL = 0.035 # Conversion from meters to pixels
    def __init__(self):
        self.map_track = TRACK # Load map track image
        self.map_grass = GRASS # Load map grass image
        pygame.display.set_caption("Map")
        self.map = pygame.display.set_mode((WIDTH,HEIGHT))
        self.show_map() # Blit map image onto display
        
        # Preset colors
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.dark_grey= (20,20,20)
        self.blue = (0, 0, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.white = (255, 255, 255)
    
    def get_pygame_surface(self):
        return self.map
        
    def pixel2position(self,pixel):
        '''
        Convert pixel into position
        '''
        posx = pixel[0]*self.METER_PER_PIXEL
        posy = pixel[1]*self.METER_PER_PIXEL
        #posy = (self.map.get_height()-pixel[1])*self.METER_PER_PIXEL
        return np.array([posx,posy])

    def position2pixel(self,position):
        '''
        Convert position into position
        '''
        pixelx = int(position[0]/self.METER_PER_PIXEL)
        pixely = int(position[1]/self.METER_PER_PIXEL)
        return np.array([pixelx,pixely])
    
    def dist2pixellen(self,dist):
        return int(dist/self.METER_PER_PIXEL)

    def show_map(self):
        '''
        Blit map onto display (Grass and Track)
        '''
        self.map.blit(self.map_grass,(0,0))
        self.map.blit(self.map_track,(0,0))
        
    def show_car(self,car, collision_points, point_pursuit):
        '''
        Blit robot onto display
        '''
        pixel_pos = self.position2pixel(car.x[:2])  # Convert car's position to pixels
        rotated_image = pygame.transform.rotate(car.car_image, -np.degrees(car.x[2]))
        new_rect = rotated_image.get_rect(center=pixel_pos)
        self.map.blit(rotated_image, new_rect.topleft)

        # Draw the ray
        self.show_ray(car)

        # Draw the collision points if they exist
        self.show_collision(collision_points)

        # Draw the point to pursuit
        #self.show_point_to_pursuit(point_pursuit)

    def show_ray(self,car):
        '''
        Blit ray onto display
        '''
        # Ray right line
        car_position_right_px = self.position2pixel(car.ray.sensor_line_right[0])
        ray_end_right_px = self.position2pixel(car.ray.sensor_line_right[1])
        pygame.draw.line(self.map, self.blue, car_position_right_px, ray_end_right_px, 1)

        # Ray left line
        car_position_left_px = self.position2pixel(car.ray.sensor_line_left[0])
        ray_end_left_px = self.position2pixel(car.ray.sensor_line_left[1])
        pygame.draw.line(self.map, self.blue, car_position_left_px, ray_end_left_px, 1)
    
    def show_collision(self, collision_points):
        '''
        Blit collision points onto display
        '''
        if collision_points[0, 0] != -1 and collision_points[0, 1] != -1:
            pygame.draw.circle(self.map, self.green, self.position2pixel(collision_points[0]), 5)
        
        if collision_points[1, 0] != -1 and collision_points[1, 1] != -1:
            pygame.draw.circle(self.map, self.green, self.position2pixel(collision_points[1]), 5)
    
    def show_point_to_pursuit(self, point_pursuit):
        '''
        Draw a point to pursuit
        '''
        if point_pursuit is not None:
            pygame.draw.circle(self.map, self.red, self.position2pixel(point_pursuit), 10)
    
    # Auxiliar function to detect collision with the window borders 
    def touch_border_window(self, ray_point):
        '''
        Detect collision of the ray with window borders
        '''
        # Define the surface's bounding rectangle
        map_rect = self.map.get_rect() 

        left, top, right, bottom = map_rect.left, map_rect.top, map_rect.right - 1, map_rect.bottom - 1

        # Define the ray_point x and y in pixels
        ray_x_px, ray_y_px = self.position2pixel(ray_point)

        # If the ray collides with the window borders, return the corresponding value (1 to left, 2 to right, 3 to top, 4 to bottom)
        if ray_x_px <= left:
            ray_x_px = left
        elif ray_x_px >= right:
            ray_x_px = right
        elif ray_y_px <= top:
            ray_y_px = top
        elif ray_y_px >= bottom:
            ray_y_px = bottom
        
        return self.pixel2position((ray_x_px, ray_y_px))

    # Auxiliar function to draw the tangent and perpendicular lines
    def _draw_tangent_and_perp(self, point, slope):
        """
        Draw a tangent and a perpendicular line to a given point
        Just to illustrate the concept and help debugging
        """
        # Tangent direction [1, slope] 
        tangent_dir = np.array([1, slope], dtype=float)
        tangent_dir /= np.linalg.norm(tangent_dir) # Normalize

        # Perpendicular direction [1, -1/slope] 
        if abs(slope) > 1e-6: # To avoid division by zero
            perpendicular_dir = np.array([1, -1.0/slope], dtype=float)
            perpendicular_dir /= np.linalg.norm(perpendicular_dir)
        else:   # slope ~ 0, then perpendicular is vertical
            perpendicular_dir = np.array([0, 1], dtype=float)

        # Size of the lines (it depends on the scale)
        line_length = 1.0 

        # Points (in meters)
        point_tangent_1 = point + line_length * tangent_dir
        point_tangent_2 = point - line_length * tangent_dir
        point_perpendicular_1 = point + line_length * perpendicular_dir
        point_perpendicular_2 = point - line_length * perpendicular_dir

        # Converter to pixel
        point_tangent_1_px, point_tangent_2_px = self.position2pixel((point_tangent_1, point_tangent_2))
        point_perpendicular_1_px, point_perpendicular_2_px = self.position2pixel((point_perpendicular_1, point_perpendicular_2))
        
        # Draw lines
        pygame.draw.line(self.map, (0, 255, 255), point_tangent_1_px,  point_tangent_2_px,  2)  # ciano
        pygame.draw.line(self.map, (255, 255, 0), point_perpendicular_1_px, point_perpendicular_2_px, 2)  # amarelo


#################### Trajectory functions ####################
def define_trajectory(car, collision_points, last20_collision_points, previous_lta, duration_lta):
    '''
    Define the trajectory of the car based on the collision points
    '''
    lta_activation, duration_lta = lta_attuation(car, last20_collision_points, previous_lta, duration_lta)

    if(lta_activation == 1):
        street_width = 11.6 # Street width in meters

        collision_right = collision_points[0]
        collision_left = collision_points[1]

        collision_right_old = last20_collision_points[-1][0]
        collision_left_old = last20_collision_points[-1][1]

        # Both Line deteted
        if collision_left[0] != -1 and collision_left[1] != -1 and collision_right[0] != -1 and collision_right[1] != -1:
            midpoint = (collision_left + collision_right) / 2.0

        # Just the left line detected 
        elif collision_left[0] != -1 and collision_left[1] != -1:
            # Calculate the slope of the tangent line (used to draw the tangent and perpendicular lines in debug)
            #m = (collision_left_old[1] - collision_left[1]) / (collision_left_old[0] - collision_left[0])
                
            # Estimate the tangent vector as the difference between next and previous points
            #tangent = np.array([collision_left[0] - collision_left_old[0] , collision_left[1] - collision_left_old[1] ])
            tangent = np.array([collision_left_old[0] - collision_left[0], collision_left_old[1] - collision_left[1]])
            tangent = tangent / np.linalg.norm(tangent)  # Normalize the tangent
            
            # Compute the normal vector (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            width = street_width / 2.0
            midpoint = [0, 0]
            midpoint[0] = collision_left[0] - width * normal[0]
            midpoint[1] = collision_left[1] - width * normal[1]

        # Just the right line detected
        elif collision_right[0] != -1 and collision_right[1] != -1:
            # Calculate the slope of the tangent line (used to draw the tangent and perpendicular lines in debug)
            #m = (collision_right_old[1] - collision_right[1]) / (collision_right_old[0] - collision_right[0])
            
            # Estimate the tangent vector as the difference between next and previous points
            #tangent = np.array([collision_right_old[0] - collision_right[0], collision_right_old[1] - collision_right[1]])
            tangent = np.array([collision_right[0] - collision_right_old[0], collision_right[1] - collision_right_old[1]])
            tangent = tangent / np.linalg.norm(tangent)  # Normalize the tangent
            
            # Compute the normal vector (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            width = street_width / 2.0
            midpoint = [0, 0]
            midpoint[0] = collision_right[0] - width * normal[0]
            midpoint[1] = collision_right[1] - width * normal[1]

        # No line detected
        else:
            midpoint = None

        # Verify if the midpoint makes sense (i.e., if it is not infinite or NaN)
        if midpoint is not None:
            if (np.isinf(midpoint[0]) or np.isinf(midpoint[1]) or
                np.isnan(midpoint[0]) or np.isnan(midpoint[1])):
                midpoint = None

        # If the midpoint is valid, return it
        if midpoint is not None:
            return midpoint, lta_activation, duration_lta

        return None, lta_activation, duration_lta
    
    else:
        return None, lta_activation, duration_lta

def pure_pursuit(car, point_pursuit, t):
    '''
    Pure pursuit algorithm
    '''
    # Calculate the delta_x and delta_y (distance between the car and the point to pursuit)
    delta_x = point_pursuit[0] - car.x[0]
    delta_y = point_pursuit[1] - car.x[1]

    alpha = np.arctan2(delta_y, delta_x) - car.x[2]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Keep angle between -pi and +pi

    distance_to_point = np.linalg.norm([delta_x, delta_y])
    
    # Calculate the steering angle (phi) based on the pure pursuit algorithm
    phi = np.arctan2(2 * car.L * np.sin(alpha), distance_to_point)
    phi = np.arctan2(np.sin(phi), np.cos(phi))  # Keep angle between -pi and +pi
    
    # Calculate the steering velocity (phi_omega) based on the new steering angle (phi)
    phi_omega = (phi - car.x[3]) / t
    phi_omega = np.arctan2(np.sin(phi_omega), np.cos(phi_omega))  # Keep angle between -pi and +pi
    
    # Keep the steering angle between -max_phi and +max_phi
    if phi > car.max_phi:
        phi = car.max_phi
    elif phi < -car.max_phi:
        phi = -car.max_phi
    
    return phi

def calculate_distances(car_position, collision_points):
    """
    Calculate the Euclidean distance from the car's position to each of the collision points.
    """
    distances = [np.linalg.norm(car_position - point) for point in collision_points]

    return np.mean(distances)

def lta_attuation(car, last20_collision_points, previous_lta, lta_duration):
    '''
    Function that will activate the lta or not (If i am to close to one of the lines, activate the lta)
    Will return lta_activation and lta_duration
    '''
    car_position_px = np.array(car.x[0], car.x[1])

    # Split into first 10 (older) and last 10 (newer) collision points
    older_left_points = [point[0] for point in last20_collision_points[:10]]
    newer_left_points = [point[0] for point in last20_collision_points[10:]]
    older_right_points = [point[1] for point in last20_collision_points[:10]]
    newer_right_points = [point[1] for point in last20_collision_points[10:]]

    # Calculate distances for older and newer points (separately for left and right)
    older_left_avg = calculate_distances(car_position_px, older_left_points)
    newer_left_avg = calculate_distances(car_position_px, newer_left_points)
    older_right_avg = calculate_distances(car_position_px, older_right_points)
    newer_right_avg = calculate_distances(car_position_px, newer_right_points)

    # Check if the average distance difference exceeds the threshold for left and right sides
    left_distance_rate = older_left_avg / newer_left_avg
    right_distance_rate = older_right_avg / newer_right_avg

    print(left_distance_rate)
    print(right_distance_rate)

    if (0.99 < abs(left_distance_rate) < 1.01 and 0.99 < abs(right_distance_rate) < 1.01): 
        lta = 0
        lta_duration = 0
        return lta, lta_duration
    elif (left_distance_rate == 1 or right_distance_rate == 1):
        lta = 1
    else: 
        lta = 1

    if previous_lta == 1 and lta_duration < 20:
        lta = 1
        lta_duration += 1

    return lta, lta_duration
    
#################### Class to deal with the plot functions ####################
# This class will be use another thread to plot the data in real time
class Plot:
    '''
    Class to deal with the plot functions
    '''
    def __init__(self):
        self.running = True # Flag to keep the plot running
        self.plot_initialization()
        #self.plot_task1_init() # Plot task 1

    def plot_initialization(self):
        '''
        Initialize the plot
        '''
        # Initialize Matplotlib for interactive mode
        plt.ion()

        # Create the figure and axis
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8,6))

        # Initialize data storage for time and detection states
        self.times = []
        self.left_detection = []
        self.right_detection = []
        self.lta = []
        self.key_left = []
        self.key_right = []

        # Create lines for left and right detection
        self.left_plot, = self.ax1.plot([], [], label="Left Detection", color="blue")
        self.right_plot, = self.ax1.plot([], [], label="Right Detection", color="red")
        self.lta_plot, = self.ax2.plot([], [], label="LTA Status", color="green")
        self.key_left_plot, = self.ax3.plot([], [], label="Key Left", color="red")
        self.key_right_plot, = self.ax3.plot([], [], label="Key Right", color="blue")

        # Configure the plot
        #self.ax1.set_title("Lane Detection Over Time")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Distance to the lines")
        self.ax1.set_ylim(0, 15)  # Fixed y-axis for detection (0 or 1)
        self.ax1.legend()

        #self.ax2.set_title("LTA Activation Over Time")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("LTA Status")
        self.ax2.set_ylim(-0.5, 1.5)  # Fixed y-axis for detection (0 or 1)
        self.ax2.legend()

        # Configure the plot
        #self.ax3.set_title("Key Pressed = 1, Key Unpressed = 0")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Key Detection")
        self.ax3.set_ylim(-0.5, 1.5)  # Fixed y-axis for detection (0 or 1)
        self.ax3.legend()
          
    def plot_update(self, t, car, collision_points, lta_activation, u):
        '''
        Continuously update the data
        '''
        distance_rigth, distance_left = self.distances_plot(car, collision_points)

        # Append the current time to the list       
        self.times.append(t)
        self.left_detection.append(distance_left)
        self.right_detection.append(distance_rigth)
        self.lta.append(1 if lta_activation else 0)

        
        if u[1] < 0:
            self.key_left.append(1)
            self.key_right.append(0)
        elif u[1] > 0:
            self.key_right.append(1)
            self.key_left.append(0)
        else:
            self.key_left.append(0)
            self.key_right.append(0)

        # Keep data within a 1-second window
        if self.times and self.times[-1] - self.times[0] > 1:
            self.times.pop(0)
            self.left_detection.pop(0)
            self.right_detection.pop(0)
            self.lta.pop(0)
            self.key_left.pop(0)
            self.key_right.pop(0)
        
        # Update plot lines
        self.left_plot.set_data(self.times, self.left_detection)
        self.right_plot.set_data(self.times, self.right_detection)
        self.lta_plot.set_data(self.times, self.lta)
        self.key_left_plot.set_data(self.times, self.key_left)
        self.key_right_plot.set_data(self.times, self.key_right)

        if self.times:
            self.ax1.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)
            self.ax2.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)
            self.ax3.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)

        
        plt.show()
        plt.pause(1e-6)
    
    def distances_plot(self, car, collision_points):
        '''
        Auxiliar function to compute the distance between the car and the lines
        '''
        car_position = car.x[:2]

        collision_right = collision_points[0]
        collision_left = collision_points[1] 

        lta_force = 0

        if collision_left[0] != -1 and collision_left[1] != -1:    
            distance_left = np.linalg.norm(collision_left - car_position)
        else:
            distance_left = np.nan

        if collision_right[0] != -1 and collision_right[1] != -1:
            distance_right = np.linalg.norm(collision_right - car_position)
        else:
            distance_right = np.nan

        return (distance_right, distance_left)

    def stop(self):
        '''
        Stop the plotting thread
        '''
        self.running = False
        plt.close(self.fig)
    
    def plot_task1_init(self):
        '''
        Init the plot to task 1
        '''
        # Initialize Matplotlib for interactive mode
        plt.ion()

        # Create the figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 6))  # 2 rows, 1 column

        # Task 1 Variables
        self.times = []
        self.phi_omega = []
        self.phi = []
        self.key_left = []
        self.key_right = []

        # Create lines for left and right detection
        self.phi_omega_plot, = self.ax3.plot([], [], label="Steering angular velocity", color="green")
        self.phi_plot, = self.ax2.plot([], [], label="Steering angule", color="purple")
        self.key_left_plot, = self.ax1.plot([], [], label="Key Left", color="red")
        self.key_right_plot, = self.ax1.plot([], [], label="Key Right", color="blue")

        # Configure the plot
        self.ax1.set_title("Key Pressed = 1, Key Unpressed = 0")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Key Detection")
        self.ax1.set_ylim(-0.5, 1.5)  # Fixed y-axis for detection (0 or 1)
        self.ax1.legend()

        #self.ax2.set_title("Steering Angule")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Steering Angule (rad)")
        self.ax2.set_ylim(-np.pi/9, np.pi/9)  # Fixed y-axis for detection (0 or 1)
        self.ax2.legend()

        #self.ax3.set_title("Steering Angular Velocity")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Steering Angular Velocity (rad/s)")
        self.ax3.set_ylim(-np.pi/4, np.pi/4)  # Fixed y-axis for detection (0 or 1)
        self.ax3.legend()


    def plot_task1(self, car, u, t):
        '''
        Plot function of the task1 demonstratation
        '''
        # Append the current time to the list       
        self.times.append(t)
        self.phi_omega.append(u[1])
        self.phi.append(car.x[3])


        if u[1] < 0:
            self.key_left.append(1)
            self.key_right.append(0)
        elif u[1] > 0:
            self.key_right.append(1)
            self.key_left.append(0)
        else:
            self.key_left.append(0)
            self.key_right.append(0)


        # Keep data within a 1-second window
        if self.times and self.times[-1] - self.times[0] > 1:
            self.times.pop(0)
            self.phi_omega.pop(0)
            self.phi.pop(0)
            self.key_left.pop(0)
            self.key_right.pop(0)
        
        # Update plot lines
        self.phi_omega_plot.set_data(self.times, self.phi_omega)
        self.phi_plot.set_data(self.times, self.phi)
        self.key_left_plot.set_data(self.times, self.key_left)
        self.key_right_plot.set_data(self.times, self.key_right)
        
        if self.times:
            self.ax1.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)
            self.ax2.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)
            self.ax3.set_xlim(max(0, self.times[-1] - 1), self.times[-1] + 0.1)
        
        plt.show()
        plt.pause(1e-6)



#################### Main function ####################
def main():
    
    clock = pygame.time.Clock()

    # Initialize and display environment
    env = Environment()

    #Initialize the plot
    plot = Plot()

    # Initialize car
    phi = 0
    theta = (1/LENGTH_CAR_AXLES) * np.tan(phi) 
    x_init = np.array([70.0, 59.2, theta, phi])
    car = Car(x_init, env)  
    
    dt = 0.01
    t = dt # Auxiliar time to plot

    u = np.array([15.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity    
    
    # Vector that saves the last 20 collisions positions
    last20_collision_points = []

    # Initial point to pursuit
    point_pursuit = [75.0, 59.2]
    
    # Vector that saves all the points to pursuit
    all_points_pursuit = []
    all_points_pursuit.append(point_pursuit)

    lta_activation = 0
    duration_lta = 0
    
    run = True
    
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                plot.stop()
                break
            u = car.update_u(u, event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states

        # Move the car
        car.move_step(u,dt,phi, lta_activation, env) # Integrate EOMs forward, i.e., move robot

        #Detect collision (border of the track) collision_points = [right, left]
        collision_points = car.ray.collide(env, TRACK_BORDER_MASK)

        # Verify if the list is empty or if the new value is different from the last stored
        if len(last20_collision_points) == 0 or (not np.array_equal(collision_points, last20_collision_points[0]) and not np.array_equal(collision_points, [(-1, -1), (-1, -1)])):
            last20_collision_points.insert(0, collision_points)
            if len(last20_collision_points) > 20:
                last20_collision_points.pop()
        
        # Define the trajectory of the car based on the collision points
        point_pursuit, lta_activation, duration_lta = define_trajectory(car, collision_points, last20_collision_points, lta_activation, duration_lta)
        if point_pursuit is not None:
            all_points_pursuit.append(point_pursuit)

        if lta_activation == 1:
            # Calculate the steering angle (phi) based on the last pure pursuit
            phi = pure_pursuit(car, all_points_pursuit[-1], dt)
        
        env.show_map() # Re-blit map
        env.show_car(car, collision_points, point_pursuit) # Re-blit car

        # Update display
        pygame.display.update() 

        #Plot data
        plot.plot_update(t, car, collision_points, lta_activation, u)
        #plot.plot_task1(car, u, t)  

        # Auxiliar print to show the position of the cursor in meters
        mouse = pygame.mouse.get_pos()
        #print(env.pixel2position(mouse))

        t += dt
        
    

if __name__ == "__main__":
    main()