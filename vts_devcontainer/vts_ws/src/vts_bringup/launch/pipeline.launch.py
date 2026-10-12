"""ROS pipeline: dataset player + graph builder, or the language node on a stored map.

    ros2 launch vts_bringup pipeline.launch.py sessions:=cold_freiburg_a_seq2_night1
    ros2 launch vts_bringup pipeline.launch.py mode:=command graph:=output/ros/cold_freiburg_a_seq2_night1/graph_0.npz \
        query:="Go to the kitchen."

Every mapping parameter comes from the vts_core configuration
(``vts_core/config/toposigma.yaml``, or ``config:=my.yaml`` merged over it), so
ROS runs and offline replays share one source of truth.
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context) -> list:
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    if arg("mode") == "command":
        return [Node(package="vts_language", executable="commands", name="commands", output="screen",
                     parameters=[{"graph_path": arg("graph"), "mode": arg("language_mode"),
                                  "query_sentence": arg("query"),
                                  "output_dir": (arg("output_dir") or os.path.dirname(arg("graph"))) + "/commands"}])]
    if arg("mode") != "building":
        raise ValueError(f"unknown mode {arg('mode')!r}")
    sessions = [s for s in arg("sessions").split(",") if s]
    output_dir = arg("output_dir") or f"output/ros/{sessions[0]}"
    player = Node(package="vts_players", executable="session_player", name="session_player", output="screen",
                  parameters=[{"sessions": sessions, "toposigma_config": arg("config"),
                               "publish_rate_hz": float(arg("player_rate_hz"))}])
    builder = Node(package="vts_mapping", executable="graph_builder", name="graph_builder", output="screen",
                   parameters=[{"toposigma_config": arg("config"), "uncertainty_contract": "path_length_budget_v1",
                                "encoder_device": arg("device"), "output_dir": output_dir}])
    # Either process ending (done or failed) takes the whole pipeline down.
    return [player, builder] + [RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=[EmitEvent(event=Shutdown(reason="pipeline node finished"))]))
        for node in (builder, player)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="building", description="building | command"),
        DeclareLaunchArgument("sessions", default_value="cold_freiburg_a_seq2_night1",
                              description="comma-separated catalogue names"),
        DeclareLaunchArgument("config", default_value="", description="YAML merged over the vts_core defaults"),
        DeclareLaunchArgument("device", default_value="auto", description="encoder device: auto|cuda|mps|cpu"),
        DeclareLaunchArgument("output_dir", default_value=""),
        DeclareLaunchArgument("player_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("graph", default_value="output/ros/graph_0.npz"),
        DeclareLaunchArgument("query", default_value="Go to the kitchen."),
        DeclareLaunchArgument("language_mode", default_value="manual", description="manual | topic"),
        OpaqueFunction(function=_nodes),
    ])
