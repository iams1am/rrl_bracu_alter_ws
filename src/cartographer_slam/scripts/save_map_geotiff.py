#!/usr/bin/env python3
"""
Save Cartographer occupancy grid map as GeoTIFF.

This script subscribes to the /map topic (OccupancyGrid) and saves it as a GeoTIFF file.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import argparse
import math
import csv
import glob
import os
import sys
import time
from pathlib import Path

from nav_msgs.msg import Path as RosPath
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    try:
        import rasterio
        from rasterio.transform import from_bounds
        HAS_RASTERIO = True
    except ImportError:
        HAS_RASTERIO = False

try:
    from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MapSaverGeoTiff(Node):
    """Node to save occupancy grid as GeoTIFF."""

    def __init__(
        self,
        output_path: str,
        map_topic: str = '/map',
        pois_csv: str = '',
        no_pois: bool = False,
        team_name: str = 'BracUAlter',
        mission: str = 'Prelim1',
        render_scale: int = 12,
        canvas_aspect: str = '16:9',
        rotate_map: str = 'ccw',
        path_wait_sec: float = 5.0,
    ):
        super().__init__('map_saver_geotiff')
        
        self.output_path = output_path
        self.pois_csv = pois_csv
        self.no_pois = no_pois
        self.team_name = team_name
        self.mission = mission
        self.render_scale = max(1, int(render_scale))
        self.canvas_aspect = self._parse_canvas_aspect(canvas_aspect)
        self.rotate_map = str(rotate_map).strip().lower()
        self.path_wait_sec = max(0.0, float(path_wait_sec))
        self.map_received = False
        self.path_received = False
        self.first_map_time = None
        self.path_points = []
        self._canvas_left = 0
        self._canvas_top = 0

        output_parent = Path(output_path).expanduser().resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info(f'Subscribing to map topic: {map_topic}')
        self.get_logger().info(f'Output will be saved to: {output_path}')
        
        self.subscription = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10
        )

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        self.path_subscription = self.create_subscription(
            RosPath,
            '/robot_path',
            self.path_callback,
            path_qos,
        )

    def path_callback(self, msg: RosPath):
        self.path_points = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self.path_received = bool(self.path_points)

    def map_callback(self, msg: OccupancyGrid):
        """Handle incoming occupancy grid message."""
        if self.map_received:
            return

        if not self.path_received:
            now = time.monotonic()
            if self.first_map_time is None:
                self.first_map_time = now
                self.get_logger().info(
                    f'Map received; waiting up to {self.path_wait_sec:.1f}s for /robot_path before export.'
                )
                return
            if now - self.first_map_time < self.path_wait_sec:
                return
            
        self.get_logger().info(
            f'Received map: {msg.info.width}x{msg.info.height}, '
            f'resolution: {msg.info.resolution}m/pixel'
        )
        
        data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )

        # Get map origin and bounds
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        resolution = msg.info.resolution
        output_resolution = resolution / self.render_scale
        
        width = msg.info.width
        height = msg.info.height
        
        # Calculate bounds
        min_x = origin_x
        max_x = origin_x + width * resolution
        min_y = origin_y
        max_y = origin_y + height * resolution
        rotate_mode = self._resolve_rotate_mode(width, height, data)
        image = self._render_rrl_map(data, msg, rotate_mode)
        min_x, max_x, min_y, max_y = self._bounds_after_rotation(
            min_x, max_x, min_y, max_y, rotate_mode
        )
        
        self.get_logger().info(
            f'Map bounds: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})'
        )
        if rotate_mode != 'none':
            self.get_logger().info(f'Rotated map geometry for horizontal export: {rotate_mode}')
        image = self._scale_for_export(image)
        image, min_x, max_x, min_y, max_y = self._fit_canvas_aspect(
            image, min_x, max_x, min_y, max_y, output_resolution
        )
        image = self._draw_rrl_overlays(image, msg, rotate_mode, output_resolution)
        self.get_logger().info(
            f'Export raster: {image.shape[1]}x{image.shape[0]} pixels, '
            f'{output_resolution:.4f}m/pixel after {self.render_scale}x render scale'
        )
        
        # Save as GeoTIFF
        success = False
        
        if HAS_GDAL:
            success = self._save_with_gdal(image, min_x, max_x, min_y, max_y, output_resolution)
        elif HAS_RASTERIO:
            success = self._save_with_rasterio(image, min_x, max_x, min_y, max_y)
        elif HAS_PIL:
            success = self._save_with_pil_geotiff(image, min_x, max_y, output_resolution)
        else:
            self.get_logger().error(
                'GDAL, rasterio, and Pillow are unavailable. '
                'Install at least one GeoTIFF/TIFF backend.'
            )
            
        if success:
            self.map_received = True
            self.get_logger().info(f'Map saved successfully to: {self.output_path}')
            
            # Also save a simple PNG for quick viewing
            png_path = self.output_path.replace('.tif', '.png').replace('.tiff', '.png')
            if png_path == self.output_path:
                png_path = self.output_path + '.png'
            self._save_png(image, png_path)
        else:
            self.get_logger().error('Failed to save map')

    def _render_rrl_map(self, data: np.ndarray, msg: OccupancyGrid, rotate_mode: str) -> np.ndarray:
        """Render the ROS occupancy grid with RRL-friendly colors."""
        source_height, source_width = data.shape
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Base RGB map in ROS grid orientation: row 0 is the map's bottom edge.
        rgb = np.zeros((source_height, source_width, 3), dtype=np.uint8)

        unknown = data == -1
        free = data == 0
        occupied = data >= 65
        uncertain = (data > 0) & (data < 65)

        # Unknown area checkerboard: 1 m squares.
        yy, xx = np.indices((source_height, source_width))
        world_x = origin_x + xx * resolution
        world_y = origin_y + yy * resolution
        checker = ((np.floor(world_x).astype(int) + np.floor(world_y).astype(int)) % 2) == 0
        rgb[unknown & checker] = (226, 226, 227)
        rgb[unknown & ~checker] = (237, 237, 238)

        # Searched/free area and probability gradient.
        rgb[free] = (255, 255, 255)
        if np.any(uncertain):
            shade = (255 - (data[uncertain].astype(np.int16) * 80 // 64)).astype(np.uint8)
            rgb[uncertain] = np.stack([shade, shade, shade], axis=1)

        # Explored area 50 cm grid, drawn behind walls/objects.
        explored = ~unknown
        grid_period = max(1, int(round(0.5 / resolution)))
        rgb[(xx % grid_period == 0) & explored] = (190, 190, 191)
        rgb[(yy % grid_period == 0) & explored] = (190, 190, 191)

        # Walls/obstacles.
        rgb[occupied] = (0, 40, 120)

        # Convert to image orientation: top-left origin.
        rgb = np.flipud(rgb)
        return self._rotate_image(rgb, rotate_mode)

    def _draw_rrl_overlays(
        self,
        image: np.ndarray,
        msg: OccupancyGrid,
        rotate_mode: str,
        output_resolution: float,
    ) -> np.ndarray:
        """Draw text, markers, path, and scale at final export resolution."""
        source_width = msg.info.width
        source_height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        height, width = image.shape[:2]

        pil_image = Image.fromarray(image, mode='RGB')
        draw = ImageDraw.Draw(pil_image)
        title_font = self._font(max(24, min(42, width // 120)), bold=True)
        label_font = self._font(max(26, min(56, width // 86)), bold=True)
        small_font = self._font(max(22, min(44, width // 100)), bold=True)

        def world_to_pixel(x_m: float, y_m: float):
            px = int(round((x_m - origin_x) / resolution))
            py = int(round(source_height - 1 - ((y_m - origin_y) / resolution)))
            px, py = self._rotate_pixel(px, py, source_width, source_height, rotate_mode)
            return (
                int(round(px * self.render_scale + self._canvas_left)),
                int(round(py * self.render_scale + self._canvas_top)),
            )

        def rotate_vector(dx: float, dy: float):
            dx, dy = self._rotate_vector(dx, dy, rotate_mode)
            return dx * self.render_scale, dy * self.render_scale

        def draw_text(pos, text, fill, font, stroke=3):
            draw.text(pos, text, fill=fill, font=font, stroke_width=stroke, stroke_fill=(245, 247, 250))

        def draw_centered_text(cx, cy, text, fill, font):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (cx - tw / 2, cy - th / 2),
                text,
                fill=fill,
                font=font,
                stroke_width=2,
                stroke_fill=(245, 247, 250),
            )

        def draw_arrow(start_x, start_y, vec_x, vec_y, color, label=None):
            length = max(1.0, math.hypot(vec_x, vec_y))
            ux = vec_x / length
            uy = vec_y / length
            end_x = start_x + vec_x
            end_y = start_y + vec_y
            line_width = max(8, int(round(self.render_scale * 0.9)))
            head_len = max(24, int(round(line_width * 3.0)))
            head_w = max(14, int(round(line_width * 1.6)))
            draw.line((start_x, start_y, end_x, end_y), fill=color, width=line_width)
            left_x = end_x - ux * head_len - uy * head_w
            left_y = end_y - uy * head_len + ux * head_w
            right_x = end_x - ux * head_len + uy * head_w
            right_y = end_y - uy * head_len - ux * head_w
            draw.polygon([(end_x, end_y), (left_x, left_y), (right_x, right_y)], fill=color)
            if label:
                draw_text((end_x + line_width, end_y + line_width), label, color, small_font, stroke=2)

        # Filename label.
        filename = Path(self.output_path).name
        draw_text((max(34, width // 40), max(32, height // 32)), filename, (0, 44, 207), title_font)

        # Scale line: exactly 1 meter.
        scale_px = max(1, int(round(1.0 / output_resolution)))
        sx2 = width - max(70, width // 26)
        sx1 = max(34, sx2 - scale_px)
        sy = max(54, height // 9)
        scale_width = max(8, int(round(self.render_scale * 0.9)))
        draw.line((sx1, sy, sx2, sy), fill=(0, 50, 140), width=scale_width)
        draw_text((sx1, sy + scale_width + 10), '1 m', (0, 50, 140), label_font, stroke=2)

        # RRL map orientation: X points upward and Y points left.
        arrow_len = max(70, int(round(0.45 / output_resolution)))
        ax = max(90, sx1 - max(130, width // 34))
        ay = sy + max(120, int(round(0.52 / output_resolution)))
        draw_arrow(ax, ay, 0, -arrow_len, (0, 50, 140), 'X')
        draw_arrow(ax, ay, -arrow_len, 0, (0, 50, 140), 'Y')

        # Initial robot pose: rules require this arrow to always point toward the top of the map.
        ox, oy = world_to_pixel(0.0, 0.0)
        if 0 <= ox < width and 0 <= oy < height:
            robot_len = max(70, int(round(0.35 / output_resolution)))
            draw_arrow(ox, oy, 0, -robot_len, (0, 240, 0))

        self._draw_robot_path(
            draw,
            world_to_pixel,
            marker_radius=max(18, min(42, int(round(0.18 / output_resolution)))),
            line_width=max(7, min(18, int(round(0.06 / output_resolution)))),
        )
        self._draw_pois(
            draw,
            world_to_pixel,
            small_font,
            marker_radius=max(24, min(56, int(round(0.22 / output_resolution)))),
            april_radius=max(22, min(52, int(round(0.175 / output_resolution)))),
        )
        return np.asarray(pil_image)

    def _font(self, size: int, bold: bool = False):
        candidates = [
            '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        for path in candidates:
            try:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _resolve_rotate_mode(self, width: int, height: int, data: np.ndarray = None) -> str:
        value = self.rotate_map
        if value in ('', 'none', 'no', 'false', '0', 'map', 'original'):
            return 'none'
        if value in ('cw', 'clockwise', 'right', '90'):
            return 'cw'
        if value in ('ccw', 'counterclockwise', 'left', '-90'):
            return 'ccw'
        if value in ('180', 'flip'):
            return '180'
        if value == 'auto':
            content_width, content_height = self._known_content_size(data, width, height)
            return 'ccw' if content_height > content_width else 'none'
        self.get_logger().warn(f'Unknown rotate_map value "{self.rotate_map}", using auto.')
        content_width, content_height = self._known_content_size(data, width, height)
        return 'ccw' if content_height > content_width else 'none'

    def _known_content_size(self, data: np.ndarray, fallback_width: int, fallback_height: int):
        if data is None:
            return fallback_width, fallback_height
        known_rows, known_cols = np.where(data != -1)
        if known_rows.size == 0 or known_cols.size == 0:
            return fallback_width, fallback_height
        content_width = int(known_cols.max() - known_cols.min() + 1)
        content_height = int(known_rows.max() - known_rows.min() + 1)
        self.get_logger().info(
            f'Known map content bounds: {content_width}x{content_height} pixels '
            f'inside {fallback_width}x{fallback_height} map'
        )
        return content_width, content_height

    def _rotate_image(self, image: np.ndarray, rotate_mode: str) -> np.ndarray:
        if rotate_mode == 'cw':
            return np.rot90(image, k=3).copy()
        if rotate_mode == 'ccw':
            return np.rot90(image, k=1).copy()
        if rotate_mode == '180':
            return np.rot90(image, k=2).copy()
        return image

    def _rotate_pixel(self, x: int, y: int, width: int, height: int, rotate_mode: str):
        if rotate_mode == 'cw':
            return height - 1 - y, x
        if rotate_mode == 'ccw':
            return y, width - 1 - x
        if rotate_mode == '180':
            return width - 1 - x, height - 1 - y
        return x, y

    def _rotate_vector(self, dx: float, dy: float, rotate_mode: str):
        if rotate_mode == 'cw':
            return -dy, dx
        if rotate_mode == 'ccw':
            return dy, -dx
        if rotate_mode == '180':
            return -dx, -dy
        return dx, dy

    def _bounds_after_rotation(self, min_x: float, max_x: float, min_y: float, max_y: float, rotate_mode: str):
        if rotate_mode not in ('cw', 'ccw'):
            return min_x, max_x, min_y, max_y
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        width_m = max_x - min_x
        height_m = max_y - min_y
        return (
            center_x - height_m / 2.0,
            center_x + height_m / 2.0,
            center_y - width_m / 2.0,
            center_y + width_m / 2.0,
        )

    def _scale_for_export(self, image: np.ndarray) -> np.ndarray:
        if self.render_scale <= 1:
            return image
        pil_image = Image.fromarray(image)
        resampling = getattr(Image, 'Resampling', Image).NEAREST
        scaled_size = (
            pil_image.width * self.render_scale,
            pil_image.height * self.render_scale,
        )
        return np.asarray(pil_image.resize(scaled_size, resampling))

    def _parse_canvas_aspect(self, canvas_aspect: str):
        value = str(canvas_aspect).strip().lower()
        if value in ('', 'none', 'map', 'original', '0'):
            return None
        if ':' in value:
            left, right = value.split(':', 1)
            return float(left) / float(right)
        return float(value)

    def _checkerboard_canvas(self, width: int, height: int) -> np.ndarray:
        yy, xx = np.indices((height, width))
        checker = ((xx // 80 + yy // 80) % 2) == 0
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[checker] = (226, 226, 227)
        canvas[~checker] = (237, 237, 238)
        return canvas

    def _fit_canvas_aspect(
        self,
        image: np.ndarray,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        resolution: float,
    ):
        self._canvas_left = 0
        self._canvas_top = 0
        if self.canvas_aspect is None:
            return image, min_x, max_x, min_y, max_y

        height, width = image.shape[:2]
        current_aspect = width / height
        target_aspect = self.canvas_aspect
        if abs(current_aspect - target_aspect) < 0.01:
            return image, min_x, max_x, min_y, max_y

        if current_aspect < target_aspect:
            new_width = int(math.ceil(height * target_aspect))
            new_height = height
        else:
            new_width = width
            new_height = int(math.ceil(width / target_aspect))

        pad_x = new_width - width
        pad_y = new_height - height
        left = pad_x // 2
        right = pad_x - left
        top = pad_y // 2
        bottom = pad_y - top
        self._canvas_left = left
        self._canvas_top = top

        canvas = self._checkerboard_canvas(new_width, new_height)
        canvas[top:top + height, left:left + width] = image

        padded_min_x = min_x - left * resolution
        padded_max_x = max_x + right * resolution
        padded_max_y = max_y + top * resolution
        padded_min_y = min_y - bottom * resolution
        self.get_logger().info(
            f'Canvas aspect padded to {target_aspect:.3f}: '
            f'{width}x{height} -> {new_width}x{new_height}'
        )
        return canvas, padded_min_x, padded_max_x, padded_min_y, padded_max_y

    def _draw_robot_path(self, draw, world_to_pixel, marker_radius: int = 6, line_width: int = 4) -> None:
        if not self.path_points:
            self.get_logger().info('No robot path available; exporting map without path overlay.')
            return
        if len(self.path_points) == 1:
            px, py = world_to_pixel(*self.path_points[0])
            radius = marker_radius
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(120, 0, 140))
            self.get_logger().info('Only one robot path pose available; drew start marker without path line.')
            return
        pixels = [world_to_pixel(x, y) for x, y in self.path_points]
        draw.line(pixels, fill=(120, 0, 140), width=line_width)

    def _latest_pois_csv(self) -> str:
        if self.no_pois:
            return ''
        if self.pois_csv:
            return self.pois_csv
        output_dir = Path(self.output_path).expanduser().resolve().parent
        candidates = sorted(glob.glob(str(output_dir / '*-pois.csv')), key=os.path.getmtime)
        return candidates[-1] if candidates else ''

    def _draw_pois(self, draw, world_to_pixel, font, marker_radius: int = 8, april_radius: int = 8) -> None:
        csv_path = self._latest_pois_csv()
        if not csv_path or not os.path.exists(csv_path):
            self.get_logger().info('No POI CSV found; exporting map without object markers.')
            return

        self.get_logger().info(f'Drawing POI markers from: {csv_path}')
        try:
            with open(csv_path, newline='') as f:
                lines = f.readlines()
        except OSError as exc:
            self.get_logger().warn(f'Could not read POI CSV: {exc}')
            return

        header_index = None
        for idx, line in enumerate(lines):
            if line.strip().startswith('detection,time,type,name,x,y,z,robot,mode'):
                header_index = idx
                break
        if header_index is None:
            return

        reader = csv.DictReader(lines[header_index:])
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
            except (KeyError, TypeError, ValueError):
                continue
            px, py = world_to_pixel(x, y)
            obj_type = row.get('type', '')
            name = row.get('name', '')
            label = row.get('detection', '').strip()
            if not label:
                label = name if obj_type == 'ar_code' else name[:2].upper()

            if obj_type == 'ar_code':
                radius = april_radius
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 200, 0))
            elif obj_type == 'hazmat_sign':
                radius = marker_radius
                draw.polygon([(px, py - radius), (px + radius, py), (px, py + radius), (px - radius, py)],
                             fill=(255, 100, 30))
            else:
                radius = marker_radius
                draw.polygon([(px, py - radius), (px + radius, py), (px, py + radius), (px - radius, py)],
                             fill=(240, 10, 10))
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (px - tw / 2, py - th / 2),
                label,
                fill=(255, 255, 255),
                font=font,
                stroke_width=max(1, marker_radius // 12),
                stroke_fill=(120, 0, 0),
            )

    def _save_with_gdal(self, image: np.ndarray, min_x: float, max_x: float, 
                        min_y: float, max_y: float, resolution: float) -> bool:
        """Save GeoTIFF using GDAL."""
        try:
            driver = gdal.GetDriverByName('GTiff')
            height, width = image.shape[:2]
            bands = 3 if image.ndim == 3 else 1
            dataset = driver.Create(
                self.output_path, 
                width, height, bands,
                gdal.GDT_Byte
            )
            
            # Set geotransform: (top_left_x, pixel_width, 0, top_left_y, 0, -pixel_height)
            geotransform = (min_x, resolution, 0, max_y, 0, -resolution)
            dataset.SetGeoTransform(geotransform)
            
            # Set projection (local coordinate system)
            srs = osr.SpatialReference()
            srs.SetLocalCS("ROS Map Coordinate System")
            dataset.SetProjection(srs.ExportToWkt())
            
            if bands == 1:
                band = dataset.GetRasterBand(1)
                band.WriteArray(image)
                band.SetNoDataValue(205)
                band.FlushCache()
            else:
                for channel in range(3):
                    band = dataset.GetRasterBand(channel + 1)
                    band.WriteArray(image[:, :, channel])
                    band.FlushCache()
            
            dataset = None  # Close file
            return True
            
        except Exception as e:
            self.get_logger().error(f'GDAL error: {e}')
            return False

    def _save_with_rasterio(self, image: np.ndarray, min_x: float, max_x: float,
                            min_y: float, max_y: float) -> bool:
        """Save GeoTIFF using rasterio."""
        try:
            height, width = image.shape[:2]
            
            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            count = 3 if image.ndim == 3 else 1
            with rasterio.open(
                self.output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=image.dtype,
                crs='+proj=longlat +datum=WGS84 +no_defs',  # Placeholder CRS
                transform=transform,
                nodata=None
            ) as dst:
                if count == 1:
                    dst.write(image, 1)
                else:
                    dst.write(np.moveaxis(image, 2, 0))
                
            return True
            
        except Exception as e:
            self.get_logger().error(f'Rasterio error: {e}')
            return False

    def _save_with_pil_geotiff(self, image: np.ndarray, min_x: float, max_y: float, resolution: float) -> bool:
        """Save TIFF with basic GeoTIFF tags using Pillow when GDAL/rasterio are unavailable."""
        try:
            img = Image.fromarray(image)
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            # GeoTIFF ModelPixelScaleTag and ModelTiepointTag.
            tags[33550] = (float(resolution), float(resolution), 0.0)
            tags[33922] = (0.0, 0.0, 0.0, float(min_x), float(max_y), 0.0)
            # Minimal GeoKeyDirectoryTag: local/user-defined projected coordinate system.
            tags[34735] = (
                1, 1, 0, 3,
                1024, 0, 1, 32767,  # GTModelTypeGeoKey = user-defined
                1025, 0, 1, 1,      # GTRasterTypeGeoKey = RasterPixelIsArea
                3072, 0, 1, 32767,  # ProjectedCSTypeGeoKey = user-defined
            )
            img.save(self.output_path, format='TIFF', tiffinfo=tags)
            return True
        except Exception as exc:
            self.get_logger().error(f'Pillow TIFF error: {exc}')
            return False

    def _save_png(self, image: np.ndarray, path: str):
        """Save a simple PNG for quick viewing."""
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img.save(path)
            self.get_logger().info(f'PNG preview saved to: {path}')
        except ImportError:
            self.get_logger().warn('PIL not available, skipping PNG export')
        except Exception as e:
            self.get_logger().warn(f'Could not save PNG: {e}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Save ROS2 map as GeoTIFF')
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='map.tif',
        help='Output GeoTIFF file path (default: map.tif)'
    )
    parser.add_argument(
        '-t', '--topic',
        type=str,
        default='/map',
        help='Map topic to subscribe to (default: /map)'
    )
    parser.add_argument(
        '--pois-csv',
        type=str,
        default='',
        help='Optional POI CSV to draw on the map. Defaults to latest *-pois.csv next to output.'
    )
    parser.add_argument(
        '--no-pois',
        action='store_true',
        help='Do not draw POI markers, even if older *-pois.csv files exist next to the output.',
    )
    parser.add_argument('--team-name', type=str, default='BracUAlter')
    parser.add_argument('--mission', type=str, default='Prelim1')
    parser.add_argument(
        '--render-scale',
        type=int,
        default=12,
        help='Integer visual upscale for exported GeoTIFF/PNG. '
             '12 creates high-resolution RRL exports while preserving map bounds.',
    )
    parser.add_argument(
        '--canvas-aspect',
        type=str,
        default='16:9',
        help='Output canvas aspect ratio, e.g. 16:9 for laptop-screen landscape. '
             'Use "map" or "none" to keep the raw SLAM map aspect.',
    )
    parser.add_argument(
        '--rotate-map',
        type=str,
        default='ccw',
        help='Rotate map geometry before export: auto, none, cw, ccw, or 180. '
             'ccw is the RRL-compliant default: map X points up and map Y points left.',
    )
    parser.add_argument(
        '--path-wait-sec',
        type=float,
        default=5.0,
        help='Seconds to wait for /robot_path after the first /map message before exporting.',
    )
    
    # Parse known args to handle ROS args
    parsed_args, remaining = parser.parse_known_args()
    
    rclpy.init(args=remaining)
    
    node = MapSaverGeoTiff(
        output_path=parsed_args.output,
        map_topic=parsed_args.topic,
        pois_csv=parsed_args.pois_csv,
        no_pois=parsed_args.no_pois,
        team_name=parsed_args.team_name,
        mission=parsed_args.mission,
        render_scale=parsed_args.render_scale,
        canvas_aspect=parsed_args.canvas_aspect,
        rotate_map=parsed_args.rotate_map,
        path_wait_sec=parsed_args.path_wait_sec,
    )
    
    try:
        # Spin until map is received
        while rclpy.ok() and not node.map_received:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
