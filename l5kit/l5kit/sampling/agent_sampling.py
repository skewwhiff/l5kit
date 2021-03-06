from typing import List, Optional, Tuple

import numpy as np

from ..data import (
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from ..data.filter import filter_agents_by_frames, filter_agents_by_track_id
from ..geometry import angular_distance, compute_agent_pose, rotation33_as_yaw, transform_point
from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from .slicing import get_future_slice, get_history_slice

def get_nearest_agent_track_ids(agents: np.ndarray, av_agent: np.ndarray, this_agent: np.ndarray, max_agents: int, agent_dist_threshold: float) -> np.ndarray:
    """
    Returns upto `max_agents` agents apart from this agent, but icluding av_agent that are within `agent_dist_threshold`
    
    """
    #add av agent too track_id = -1
    track_ids = agents["track_id"]
    track_ids = np.append(track_ids, -1) #-1 is to signify av_agent

    centroids = agents["centroid"]
    centroids = np.vstack((centroids, av_agent["ego_translation"][:2]))
    
    dists = np.linalg.norm(centroids - this_agent["centroid"], axis=1)
    indices = dists.argsort()[:max_agents+1]
    indices = indices[dists[indices]<=agent_dist_threshold]
    nearest_agent_track_ids = track_ids[indices].tolist()
    return nearest_agent_track_ids, indices

def generate_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames: int,
    history_step_size: int,
    history_step_time: float,
    future_num_frames: int,
    future_step_size: int,
    future_step_time: float,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    agent_dist_threshold: Optional[float] = 20.,
    max_agents: Optional[int] = 4,
    lane_dist_threshold: Optional[float] = 17.,
    is_world_frame: Optional[bool] = False,
    return_image: Optional[bool] = False,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.
    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.
    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size (Tuple[int, int]): Desired output raster dimensions
        pixel_size (np.ndarray): Size of one pixel in the real world
        ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        is_world_frame (Optional[bool]): Is output in world frame? If true, outputs are in world frame. Else, outputs are in agent frame. Rest should be self explanatory. 
        return_image (Optional[bool]): Return image too? CAUTION: Will repeat the same ops twice. 
    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.
    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
        nearest_agent_track_ids = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            useful_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
            agent = filter_agents_by_track_id(
                useful_agents, selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent 
        nearest_agent_track_ids, nearest_agent_indices = get_nearest_agent_track_ids(useful_agents, cur_frame, selected_agent, max_agents, agent_dist_threshold)

    input_im = (
        None
        if not rasterizer or not return_image
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )
    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    transformer = agent_from_world if not is_world_frame else np.eye(agent_from_world.shape[0], agent_from_world.shape[1])
    
    lane_centerlines, lane_traffic_status, lane_probabilities, crosswalks = None, None, None, None
    if hasattr(rasterizer, 'sem_rast'):
        _, lane_centerlines, lane_traffic_status, lane_probabilities, crosswalks = rasterizer.sem_rast.get_raw_data(agent_centroid_m, history_tl_faces[0], lane_dist_threshold, transformer) 
    
    future_positions_m, future_yaws_rad, future_availabilities = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, [selected_track_id] if selected_track_id else None, future_agents, transformer, agent_yaw_rad
    )
    future_positions_m, future_yaws_rad, future_availabilities = future_positions_m[0], future_yaws_rad[0], future_availabilities[0] #sindhu NOTE: Only this agent's inputs are required for future.

    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_availabilities = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, nearest_agent_track_ids, history_agents, transformer, agent_yaw_rad
    )

    # compute estimated velocities by finite differentiatin on future positions
    # estimate velocity at T with (pos(T+t) - pos(T))/t
    # this gives < 0.5% velocity difference to (pos(T+t) - pos(T-t))/2t on v1.1/sample.zarr.tar

    # [future_num_frames, 2]
    future_positions_diff_m = np.concatenate((future_positions_m[:1], np.diff(future_positions_m, axis=0)))
    # [future_num_frames, 2]
    future_vels_mps = np.float32(future_positions_diff_m / future_step_time)

    # current position is included in history positions
    # [history_num_frames, 2]
    history_positions_diff_m = [np.diff(hpos, axis=0) for hpos in history_positions_m]
    # [history_num_frames, 2]
    history_vels_mps = [np.float32(hpos_diff_m / history_step_time) for hpos_diff_m in history_positions_diff_m]

    return {
        "image": input_im,
        "lanes": lane_centerlines,
        "lane_traffic_status": lane_traffic_status, 
        "lane_probabilities": lane_probabilities,
        "crosswalks": crosswalks, 
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m[0],
        "history_yaws": history_yaws_rad[0],
        "history_velocities": history_vels_mps[0],
        "history_availabilities": history_availabilities[0],
        "nearest_agents_relative_indices": nearest_agent_indices, #sindhu NOTE needed for to_rgb. Also index of av = -1.  
        "other_agents_history_positions": history_positions_m[1:],
        "other_agents_history_yaws": history_yaws_rad[1:],
        "other_agents_history_velocities": history_vels_mps[1:],
        "other_agents_history_availabilities": history_availabilities[1:],
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,        
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm(future_vels_mps[0]),
        "extent": agent_extent_m,
    }

def _create_targets_for_deep_prediction(
    num_frames: int,
    frames: np.ndarray,
    selected_track_ids: Optional[List[int]],
    agents: List[np.ndarray],
    agent_from_world: np.ndarray,
    current_agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).
    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_ids (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities
    """
    # How much the coordinates differ from the current state in meters.
    positions_m = [np.zeros((num_frames, 2), dtype=np.float32) for _ in selected_track_ids] if selected_track_ids is not None else [np.zeros((num_frames, 2), dtype=np.float32)]
    yaws_rad = [np.zeros((num_frames, 1), dtype=np.float32) for _ in selected_track_ids] if selected_track_ids is not None else [np.zeros((num_frames, 1), dtype=np.float32)]
    availabilities = [np.zeros((num_frames,), dtype=np.float32) for _ in selected_track_ids] if selected_track_ids is not None else [np.zeros((num_frames,), dtype=np.float32)]

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_ids is None:
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
            positions_m[0][i] = transform_point(agent_centroid_m, agent_from_world)
            yaws_rad[0][i] = angular_distance(agent_yaw_rad, current_agent_yaw)
            availabilities[0][i] = 1.0
        else:
            for t_ix, f_track_id in enumerate(selected_track_ids):
                # it's not guaranteed the target will be in every frame
                filtered_agents = filter_agents_by_track_id(frame_agents, f_track_id)
                if len(filtered_agents)>0:
                    agent = filtered_agents[0]
                    agent_centroid_m = agent["centroid"]
                    agent_yaw_rad = agent["yaw"]
                    #sindhu TODO: Send dist also?
                    positions_m[t_ix][i] = transform_point(agent_centroid_m, agent_from_world)
                    yaws_rad[t_ix][i] = angular_distance(agent_yaw_rad, current_agent_yaw)
                    availabilities[t_ix][i] = 1.0
                elif f_track_id == -1: #AV is in agent's radius
                    agent_centroid_m = frame["ego_translation"][:2]
                    agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
                    positions_m[t_ix][i] = transform_point(agent_centroid_m, agent_from_world)
                    yaws_rad[t_ix][i] = angular_distance(agent_yaw_rad, current_agent_yaw)
                    availabilities[t_ix][i] = 1.0

    return positions_m, yaws_rad, availabilities
