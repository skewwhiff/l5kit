from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..data.filter import filter_tl_faces_by_status
from ..data.map_api import MapAPI
from ..geometry import rotation33_as_yaw, transform_point, transform_points
from .rasterizer import Rasterizer
from .render_context import RenderContext

# sub-pixel drawing precision constants
CV2_SHIFT = 8  # how many bits to shift in drawing
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT


def elements_within_bounds(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center <= bounds[:, 1, 0] + half_extent
    y_max_in = y_center <= bounds[:, 1, 1] + half_extent
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords


class SemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
        self, render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray,
    ):
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        self.proto_API = MapAPI(semantic_map_path, world_to_ecef)

        self.bounds_info = self.get_bounds()

    # TODO is this the right place for this function?
    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []

        lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

        for element in self.proto_API:
            element_id = MapAPI.id_as_str(element.id)

            if self.proto_API.is_lane(element):
                lane = self.proto_API.get_lane_coords(element_id)
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_ids.append(element_id)

            if self.proto_API.is_crosswalk(element):
                crosswalk = self.proto_API.get_crosswalk_coords(element_id)
                x_min = np.min(crosswalk["xyz"][:, 0])
                y_min = np.min(crosswalk["xyz"][:, 1])
                x_max = np.max(crosswalk["xyz"][:, 0])
                y_max = np.max(crosswalk["xyz"][:, 1])

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                crosswalks_ids.append(element_id)

        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
        }

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_im.astype(np.float32) / 255

    def get_raw_data(
            self, center_in_world: np.ndarray, tl_faces: np.ndarray, raster_radius: float, transformer_matrix: np.ndarray, lane_smooth_probability: Optional[float]=None
    ) -> Tuple[List[str], List[np.ndarray], List[str], List[float], List[np.ndarray]]:
        """Renders raw data in world coordinates

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            transformer_matrix (np.ndarray): 3x3 matrix to transform.
            tl_faces: Traffic light controlling lanes and crosswalks
            raster_radius: Radius to search lanes and crosswalks
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Lane raw data, crosswalks raw data with smoothed probabilities if lane_smooth_probabilities is not none
        """
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())
        nearby_lane_indices = elements_within_bounds(center_in_world, self.bounds_info["lanes"]["bounds"], raster_radius)
        candidate_lane_indices = elements_within_bounds(center_in_world, self.bounds_info["lanes"]["bounds"], 0)
        
        #1. candidate_lane_indices shape = 0, nearby_lane_indices shape > 0, lane probabilities of all lanes = 1/nearby lane indices shape
        #2. candidate lane indices shape =0, nearby lane indices shape = 0, list is empty
        #3. candidate_lane_indices shape =  nearby_lane_indices != 0, lane probabilities = 1/candidate lanes
        #4. else for candidates: label smooth/total candidates, for other lanes: (1-label smooth)/(total nearby lanes - total candidates)

        lane_ids = list()
        lane_cls = list()
        lane_traffic_types = list()
        lane_probabilities = list()
        for idx in nearby_lane_indices:
            lane_id = self.bounds_info["lanes"]["ids"][idx]
            lane_ids.append(lane_id)
            lane = self.proto_API[lane_id].element.lane

            #centerline
            lane_cl = self.proto_API.get_lane_coords(lane_id)
            lane_cl = lane_cl["xyz"]
            lane_cl = transform_points(lane_cl, transformer_matrix)
            lane_cls.append(lane_cl)

            #traffic light status
            lane_type = "default"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"
            lane_traffic_types.append(lane_type)

            #lane probability
            lane_probability = 0.0
            if lane_smooth_probability:
                #sindhu NOTE: No need of denominator 0 checks if we are looping through one at a time. 
                if idx in candidate_lane_indices:
                    lane_probability = lane_smooth_probability/candidate_lane_indices.shape[0]
                else:
                    lane_probability = (1-lane_smooth_probability)/(nearby_lane_indices.shape[0]-candidate_lane_indices.shape[0])
            lane_probabilities.append(lane_probability)

        crosswalks = []
        for idx in elements_within_bounds(center_in_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])
            crosswalk = crosswalk["xyz"]
            crosswalk = transform_points(crosswalk[:, :2], transformer_matrix)
            crosswalks.append(crosswalk)

        return lane_ids, lane_cls, lane_traffic_types, lane_probabilities, crosswalks

    def render_semantic_map(
        self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get raw data
        lane_ids, lane_cls, lane_traffic_types, lane_probabilities, crosswalks = self.get_raw_data(center_in_world, tl_faces, raster_radius, raster_from_world, 1.)

        # plot lanes
        lanes_lines = defaultdict(list)
        lane_traffic_keys = list(set(lane_traffic_types))
        lane_prob_keys = sorted(list(set(lane_probabilities)))
        lanes_lines = {(lt, lp): [] for lt in lane_traffic_keys for lp in lane_prob_keys}

        for lane_id, lane_cl, ltraffic, lprob in zip(lane_ids, lane_cls, lane_traffic_types, lane_probabilities):
            lane_coords = self.proto_API.get_lane_coords(lane_id)
            xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
            xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))
            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

            # Note(lberg): this called on all polygons skips some of them, don't know why
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            lanes_lines[(ltraffic, lprob)].append(cv2_subpixel(lane_cl))
        
        lane_traffic_lookup = {"default": (255, 217, 82), "green": (0, 255, 0), "yellow": (255, 255, 0), "red": (255, 0, 0)}
        lane_prob_lookup  = {k : ix+1 for ix, k in enumerate(lane_prob_keys)}
        for (ltraffic, lprob), cls in lanes_lines.items():
            cv2.polylines(img, cls, False, lane_traffic_lookup[ltraffic], lane_prob_lookup[lprob], lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # plot crosswalks
        cv2.polylines(img, [cv2_subpixel(cw) for cw in crosswalks], True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)

