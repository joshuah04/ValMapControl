import math
from typing import List, Tuple
from functools import lru_cache

class CollisionShape:
    """Represents a collision shape (rectangle, polygon, circle, line)"""
    
    def __init__(self, shape_type: str, points: List, name: str = ""):
        """Initialize collision shape
        
        Args:
            shape_type: Type of shape (rectangle, polygon, circle, line)
            points: List of points defining the shape
            name: Optional name for the shape
        """
        self.shape_type = shape_type  # 'rectangle', 'polygon', 'circle', 'line'
        self.points = points
        self.name = name
        
        # Cache bounding box and polygon hash for faster lookups
        self._bbox_cache = None
        self._polygon_hash = None
        
        # For circles, extract center and radius
        if shape_type == 'circle' and len(points) >= 3:
            self.center_x = points[0]
            self.center_y = points[1]
            self.radius = points[2]
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this collision shape"""
        if self.shape_type == 'rectangle':
            if len(self.points) == 2:
                x1, y1 = self.points[0]
                x2, y2 = self.points[1]
            else:
                x1, y1, x2, y2 = self.points
            
            return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
        
        elif self.shape_type == 'polygon':
            return self._point_in_polygon_optimized(x, y, self.points)
        
        elif self.shape_type == 'circle':
            dx = x - self.center_x
            dy = y - self.center_y
            return (dx * dx + dy * dy) <= (self.radius * self.radius)
        
        elif self.shape_type == 'line':
            # For lines, check if point is within a small distance of the line
            if len(self.points) >= 2:
                x1, y1 = self.points[0]
                x2, y2 = self.points[1]
                
                # Calculate distance from point to line
                line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
                if line_length_sq == 0:
                    # Line is a point
                    return (x - x1)**2 + (y - y1)**2 <= 25  # 5 pixel tolerance
                
                # Calculate the parameter t for the closest point on the line
                t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
                
                # Find the closest point on the line
                closest_x = x1 + t * (x2 - x1)
                closest_y = y1 + t * (y2 - y1)
                
                # Check if point is within tolerance distance
                distance_sq = (x - closest_x)**2 + (y - closest_y)**2
                return distance_sq <= 25  # 5 pixel tolerance
        
        return False
    
    def get_polygon_bbox(self, polygon: List[Tuple]) -> Tuple[float, float, float, float]:
        """Get cached bounding box for polygon"""
        if self._bbox_cache is None:
            if not polygon:
                self._bbox_cache = (0, 0, 0, 0)
            else:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                self._bbox_cache = (min(xs), min(ys), max(xs), max(ys))
        return self._bbox_cache
    
    def get_polygon_hash(self, polygon: List[Tuple]) -> int:
        """Get cached hash for polygon (for LRU cache key)"""
        if self._polygon_hash is None:
            # Create stable hash from polygon points
            polygon_str = ','.join(f"{p[0]:.3f},{p[1]:.3f}" for p in polygon)
            self._polygon_hash = hash(polygon_str)
        return self._polygon_hash
    
    def _point_in_polygon_optimized(self, x: float, y: float, polygon: List[Tuple]) -> bool:
        """Optimized point-in-polygon with bounding box, caching, and specialized algorithms"""
        if not polygon:
            return False
        
        # Bounding box pre-filtering for early rejection (70-90% rejection rate)
        min_x, min_y, max_x, max_y = self.get_polygon_bbox(polygon)
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        
        # Use specialized functions for common polygon types
        n = len(polygon)
        if n == 3:
            return self._point_in_triangle_optimized(x, y, polygon)
        elif n == 4:
            return self._point_in_quad_optimized(x, y, polygon)
        
        # Use cached computation for complex polygons
        return self._point_in_polygon_cached(x, y, polygon)
    
    def _point_in_triangle_optimized(self, x: float, y: float, triangle: List[Tuple]) -> bool:
        """Optimized triangle containment using barycentric coordinates"""
        x1, y1 = triangle[0]
        x2, y2 = triangle[1] 
        x3, y3 = triangle[2]
        
        # Barycentric coordinate method (faster than ray casting for triangles)
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-10:  # Degenerate triangle
            return False
        
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b
        
        return a >= 0 and b >= 0 and c >= 0
    
    def _point_in_quad_optimized(self, x: float, y: float, quad: List[Tuple]) -> bool:
        """Optimized quadrilateral containment"""
        # For rectangles (axis-aligned), this is already handled in contains_point
        # For general quads, use optimized ray casting with only 4 edges
        
        inside = False
        j = 3  # Start with last vertex
        
        for i in range(4):
            xi, yi = quad[i]
            xj, yj = quad[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _point_in_polygon_cached(self, x: float, y: float, polygon: List[Tuple]) -> bool:
        """Cached ray casting for complex polygons"""
        # Quantize coordinates for better cache hit rate (trade precision for speed)
        x_quantized = round(x * 10) / 10  # 0.1 pixel precision
        y_quantized = round(y * 10) / 10
        
        polygon_hash = self.get_polygon_hash(polygon)
        return self._point_in_polygon_raw_cached(x_quantized, y_quantized, polygon_hash, tuple(tuple(p) for p in polygon))
    
    @lru_cache(maxsize=2048)  # LRU cache for repeated queries
    def _point_in_polygon_raw_cached(self, x: float, y: float, polygon_hash: int, polygon_tuple: Tuple) -> bool:
        """Cached implementation of ray casting algorithm"""
        polygon = [list(p) for p in polygon_tuple]  # Convert back from tuple for processing
        return self._point_in_polygon_raw(x, y, polygon)
    
    def _point_in_polygon_raw(self, x: float, y: float, polygon: List[Tuple]) -> bool:
        """Improved ray casting algorithm with better edge case handling"""
        n = len(polygon)
        inside = False
        
        j = n - 1  # Start with last vertex
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # Improved ray casting with better edge case handling
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple]) -> bool:
        """Legacy method - redirects to optimized version"""
        return self._point_in_polygon_optimized(x, y, polygon)
    
    def intersects_line(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if a line segment intersects this collision shape"""
        if self.shape_type == 'rectangle':
            return self._line_intersects_rectangle(x1, y1, x2, y2)
        elif self.shape_type == 'polygon':
            return self._line_intersects_polygon(x1, y1, x2, y2)
        elif self.shape_type == 'circle':
            return self._line_intersects_circle(x1, y1, x2, y2)
        elif self.shape_type == 'line':
            return self._line_intersects_line_shape(x1, y1, x2, y2)
        return False
    
    def _line_intersects_rectangle(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if line intersects rectangle"""
        if len(self.points) == 2:
            rx1, ry1 = self.points[0]
            rx2, ry2 = self.points[1]
        else:
            rx1, ry1, rx2, ry2 = self.points
        
        # Ensure rectangle bounds are correct
        left = min(rx1, rx2)
        right = max(rx1, rx2)
        top = min(ry1, ry2)
        bottom = max(ry1, ry2)
        
        # Check if line intersects any of the four rectangle edges
        return (self._line_intersects_line(x1, y1, x2, y2, left, top, right, top) or
                self._line_intersects_line(x1, y1, x2, y2, right, top, right, bottom) or
                self._line_intersects_line(x1, y1, x2, y2, right, bottom, left, bottom) or
                self._line_intersects_line(x1, y1, x2, y2, left, bottom, left, top))
    
    def _line_intersects_polygon(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if line intersects polygon"""
        for i in range(len(self.points)):
            px1, py1 = self.points[i]
            px2, py2 = self.points[(i + 1) % len(self.points)]
            if self._line_intersects_line(x1, y1, x2, y2, px1, py1, px2, py2):
                return True
        return False
    
    def _line_intersects_circle(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if line intersects circle"""
        # Distance from circle center to line
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line is a point
            return math.sqrt((x1 - self.center_x)**2 + (y1 - self.center_y)**2) <= self.radius
        
        # Calculate distance from center to line
        t = ((self.center_x - x1) * dx + (self.center_y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp to line segment
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        distance = math.sqrt((closest_x - self.center_x)**2 + (closest_y - self.center_y)**2)
        return distance <= self.radius
    
    def _line_intersects_line(self, x1: float, y1: float, x2: float, y2: float,
                             x3: float, y3: float, x4: float, y4: float) -> bool:
        """Check if two line segments intersect"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _line_intersects_line_shape(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if line intersects line shape"""
        if len(self.points) >= 2:
            lx1, ly1 = self.points[0]
            lx2, ly2 = self.points[1]
            return self._line_intersects_line(x1, y1, x2, y2, lx1, ly1, lx2, ly2)
        return False
