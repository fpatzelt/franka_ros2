// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <realtime_tools/realtime_buffer.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"

#include <controller_interface/controller_interface.hpp>
#include <franka_msgs/srv/set_cartesian_stiffness.hpp>
#include "franka_semantic_components/franka_cartesian_pose_interface.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"

#include "franka_example_controllers/visibility_control.h"

namespace franka_example_controllers {

class CartesianImpedanceExampleController : public controller_interface::ControllerInterface {
 public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

 private:
  struct TargetPose {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{1.0, 0.0, 0.0, 0.0};
    /// Counts the received targets, so update() can tell a new one from a repeated read.
    std::uint64_t sequence{0};
  };

  static constexpr int num_joints{7};
  static constexpr int num_cartesian_dof{6};

  struct CartesianGains {
    Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> stiffness{
        Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof>::Zero()};
    Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> damping{
        Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof>::Zero()};
  };

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;

  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  static constexpr bool k_elbow_activated_{false};

  std::string arm_id_;
  std::string arm_prefix_;

  Eigen::Matrix<double, num_joints, 1> q_{Eigen::Matrix<double, num_joints, 1>::Zero()};
  Eigen::Matrix<double, num_joints, 1> dq_{Eigen::Matrix<double, num_joints, 1>::Zero()};

  double filter_params_{0.005};
  /// Set from the ``filter_params`` parameter and re-readable at runtime; see onParameterUpdate.
  realtime_tools::RealtimeBuffer<double> filter_params_buffer_;

  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> cartesian_stiffness_;
  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> cartesian_damping_;
  double nullspace_stiffness_{20.0};

  Eigen::Vector3d position_d_;
  Eigen::Vector3d initial_position_;
  double elapsed_time_{0.0};
  Eigen::Quaterniond orientation_d_;

  Eigen::Matrix<double, num_joints, 1> q_d_nullspace_;

  realtime_tools::RealtimeBuffer<TargetPose> target_pose_buffer_;
  realtime_tools::RealtimeBuffer<CartesianGains> cartesian_gains_buffer_;
  realtime_tools::RealtimeBuffer<double> nullspace_stiffness_buffer_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_equilibrium_pose_;
  rclcpp::Service<franka_msgs::srv::SetCartesianStiffness>::SharedPtr srv_set_cartesian_stiffness_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  /// @brief Callback for incoming equilibrium pose messages.
  /// @param msg New target pose message.
  void equilibriumPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  /// @brief Callback for setting Cartesian stiffness via service.
  /// @param request Service request containing 6 stiffness values.
  /// @param response Service response indicating success or failure.
  void setCartesianStiffnessCallback(
      const std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Request> request,
      std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Response> response);

  /// @brief Callback for live parameter updates: nullspace_stiffness, filter_params,
  /// damping_ratio, translational_inertia, rotational_inertia, velocity_feedforward,
  /// acceleration_feedforward, interpolate_target, joint_torque_offset, the friction settings
  /// and the integral_* parameters.
  /// @param parameters List of updated parameters.
  rcl_interfaces::msg::SetParametersResult onParameterUpdate(
      const std::vector<rclcpp::Parameter>& parameters);

  /// @brief Reads joint positions and velocities from loaned state interfaces.
  void readJointState();

  /// @brief Updates the internal demo motion target in the RT buffer.
  /// @param period_seconds Time elapsed since last update cycle.
  /// @param orientation Current end-effector orientation to maintain.
  void updateMotionTarget(double period_seconds, const Eigen::Quaterniond& orientation);

  /// @brief Computes the 6D Cartesian pose error between current and desired pose.
  /// @param position Current end-effector position.
  /// @param orientation Current end-effector orientation.
  /// @param transform Current end-effector transform.
  /// @return 6D error vector [position_error; orientation_error].
  Eigen::Matrix<double, num_cartesian_dof, 1> computeError(const Eigen::Vector3d& position,
                                                           const Eigen::Quaterniond& orientation,
                                                           const Eigen::Affine3d& transform) const;

  /// Rate, in 1/s, at which the nullspace target follows the measured joint positions: each
  /// cycle it moves min(1, rate * dt) of the way to them, which keeps the joint error the
  /// posture term pulls against small at any tool position. 0.0 pins the target at its
  /// activation value.
  double nullspace_follow_rate_{1.0};
  /// Damping ratio and the Cartesian inertias, in kg and kg m^2, that buildGains designs the
  /// damping for. Written by the parameter callback; update() reads the inertias when
  /// inertia_shaping is on.
  std::atomic<double> damping_ratio_{1.0};
  std::atomic<double> translational_inertia_{3.0};
  std::atomic<double> rotational_inertia_{0.25};
  /// The stiffness last asked for, which a change of the damping design is rebuilt against.
  /// Guarded by gains_mutex_, as is every write to the gains buffer that depends on it.
  std::array<double, num_cartesian_dof> stiffness_target_{};
  std::mutex gains_mutex_;

  /// Per-joint Coulomb friction the arm is believed to have, in Nm, fed forward as the
  /// FrictionSettings below describe.
  std::array<double, num_joints> friction_torque_{};
  /// How much of `friction_torque_` is fed forward, and along which velocity. All four are
  /// settable at run time.
  struct FrictionSettings {
    /// Fraction applied along the measured joint velocity. Compensating a friction over-estimate
    /// along the measured velocity is negative damping at low speed and drives a limit cycle.
    double compensation{0.0};
    /// Velocity scale, in rad/s, of the tanh that gives the compensation the sign of dq. Below
    /// it the compensation tapers off, so it changes sign smoothly as dq crosses zero.
    double velocity_eps{0.01};
    /// Fraction applied along the joint velocity the moving target asks for, J^+ times the
    /// feedforward twist. It does not depend on the measured motion, so it cannot feed back.
    /// The two fractions together may not exceed 1.
    double target_compensation{0.0};
    /// Velocity scale, in rad/s, of the tanh on the target's joint velocity.
    double target_velocity_eps{0.01};
  };
  static constexpr std::array<std::pair<const char*, double FrictionSettings::*>, 4>
      k_friction_parameters_{{
          {"friction_compensation", &FrictionSettings::compensation},
          {"friction_velocity_eps", &FrictionSettings::velocity_eps},
          {"friction_target_compensation", &FrictionSettings::target_compensation},
          {"friction_target_velocity_eps", &FrictionSettings::target_velocity_eps},
      }};
  FrictionSettings friction_settings_;
  realtime_tools::RealtimeBuffer<FrictionSettings> friction_buffer_;

  /// Constant torque per joint, in Nm, added to the command: the part of the arm's statics the
  /// robot model does not predict. Settable at run time, at most k_max_joint_torque_offset_ per
  /// joint.
  realtime_tools::RealtimeBuffer<std::array<double, num_joints>> joint_torque_offset_buffer_;
  static constexpr double k_max_joint_torque_offset_{2.0};

  /// Integral action on the Cartesian error. It accumulates a wrench only while the filtered
  /// target and the tool are both nearly still, and decays at the same rate while either moves.
  struct IntegralSettings {
    /// Gain as a fraction of the stiffness, per second: each settled cycle adds
    /// `-rate * K * e * dt`, so a steady offset decays with a time constant of 1/rate.
    /// 0 disables the term and empties the accumulator.
    double rate{0.0};
    /// Largest accumulated force, in N, and moment, in Nm. Each half is scaled down to its limit.
    double force_limit{1.0};
    double moment_limit{0.3};
    /// Speeds, in m/s and rad/s. The accumulator integrates while the filtered target and the
    /// tool both move slower than these, and decays otherwise.
    double max_speed{0.01};
    double max_angular_speed{0.05};
    /// Position error in m and rotation angle in rad that is not integrated; only the part
    /// beyond it is.
    double position_deadband{0.0002};
    double orientation_deadband{0.002};
  };
  /// Parameter name of every IntegralSettings field. Declared, validated and settable at run
  /// time through this table.
  static constexpr std::array<std::pair<const char*, double IntegralSettings::*>, 7>
      k_integral_parameters_{{
          {"integral_rate", &IntegralSettings::rate},
          {"integral_force_limit", &IntegralSettings::force_limit},
          {"integral_moment_limit", &IntegralSettings::moment_limit},
          {"integral_max_speed", &IntegralSettings::max_speed},
          {"integral_max_angular_speed", &IntegralSettings::max_angular_speed},
          {"integral_position_deadband", &IntegralSettings::position_deadband},
          {"integral_orientation_deadband", &IntegralSettings::orientation_deadband},
      }};
  /// Written by on_configure and the parameter callback, copied into the buffer after each
  /// change.
  IntegralSettings integral_settings_;
  realtime_tools::RealtimeBuffer<IntegralSettings> integral_buffer_;
  /// The accumulated wrench, base frame, force then moment.
  Eigen::Matrix<double, num_cartesian_dof, 1> integral_wrench_{
      Eigen::Matrix<double, num_cartesian_dof, 1>::Zero()};
  /// Velocity of the filtered target over the last cycle, base frame, linear then angular.
  Eigen::Matrix<double, num_cartesian_dof, 1> target_twist_{
      Eigen::Matrix<double, num_cartesian_dof, 1>::Zero()};
  /// target_twist_ passed through a second first-order filter with the coefficient
  /// filter_params, which smooths the steps a streamed target makes at its update rate.
  Eigen::Matrix<double, num_cartesian_dof, 1> target_twist_filtered_{
      Eigen::Matrix<double, num_cartesian_dof, 1>::Zero()};
  /// Change of target_twist_ per second, base frame, passed through a first-order filter with
  /// the coefficient filter_params.
  Eigen::Matrix<double, num_cartesian_dof, 1> target_acceleration_{
      Eigen::Matrix<double, num_cartesian_dof, 1>::Zero()};
  /// Fraction of the feedforward twist the damping is measured against, from 0 to 1. At 0 the
  /// damping acts on the tool's absolute velocity and the tool trails a moving target by D/K; at
  /// 1 it acts on the velocity relative to the target and that trailing error is gone. The
  /// feedforward twist is target_twist_ while the target is interpolated and
  /// target_twist_filtered_ otherwise. Settable at run time.
  std::atomic<double> velocity_feedforward_{0.0};
  /// Fraction of the task inertia times target_acceleration_ added to the wrench, from 0 to 1:
  /// the force the arm needs to follow the target's acceleration. It acts only while the target
  /// is interpolated. The inertia is the one built with shaping_damping, and the result is scaled
  /// to shaping_force_limit and shaping_moment_limit, which must both be positive for it.
  /// Settable at run time.
  std::atomic<double> acceleration_feedforward_{0.0};

  /// When set, each received target is reached along a straight line, in position and by slerp
  /// in orientation, over the mean interval at which targets arrive, starting from where the
  /// reference is when it arrives. The target filter follows that reference; when unset, the
  /// reference steps to each received target. Settable at run time.
  std::atomic<bool> interpolate_target_{false};
  /// Written by the subscription callback.
  std::atomic<std::uint64_t> target_sequence_{0};
  /// Sequence of the last target update() has seen.
  std::uint64_t seen_sequence_{0};
  /// The reference the target filter follows, and the segment it is moving along.
  Eigen::Vector3d reference_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond reference_orientation_{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d segment_start_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond segment_start_orientation_{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d segment_end_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond segment_end_orientation_{1.0, 0.0, 0.0, 0.0};
  double segment_elapsed_{0.0};
  /// Mean interval between received targets, in s, and the time since the last one.
  double arrival_interval_{0.02};
  double since_arrival_{1.0};
  /// Whether an interval has been measured since activation; the first one is taken as the mean.
  bool arrival_interval_measured_{false};
  /// Intervals outside this range, in s, do not update the mean; it moves by
  /// k_arrival_smoothing_ of the difference per arrival.
  static constexpr double k_min_arrival_interval_{0.005};
  static constexpr double k_max_arrival_interval_{0.1};
  static constexpr double k_arrival_smoothing_{0.1};

  /// How far to cancel the arm's own task-space inertia, from 0 (not at all) to 1 (fully). The
  /// wrench is premultiplied by `(1 - s) I + s Lambda Lambda_d^-1`, with Lambda_d the diagonal of
  /// translational_inertia_ and rotational_inertia_. At 1 a commanded wrench accelerates the
  /// tool as it would accelerate Lambda_d, in every posture.
  double inertia_shaping_{0.0};
  /// When set, the posture torque is projected with the dynamically consistent
  /// `I - J^T Lambda J M^-1`; otherwise with `I - J^T (J^T)^+`, where (J^T)^+ is the damped
  /// pseudo-inverse. The damped projector lets part of the posture torque reach the tool.
  bool consistent_nullspace_{false};
  /// Damping lambda in `Lambda = (J M^-1 J^T + lambda^2 I)^-1` for the consistent projector.
  /// The larger it is, the more posture torque reaches the tool; the smaller, the worse the
  /// inversion is conditioned near a singularity.
  double task_inertia_damping_{0.01};
  /// The same damping for the Lambda that inertia shaping and acceleration feedforward use. It
  /// bounds Lambda near a singularity, where the undamped Lambda grows without limit and scales
  /// the wrench with it.
  double shaping_damping_{0.2};
  /// Largest norm, in N, of the force part of the shaped wrench and of the acceleration
  /// feedforward. A larger force is scaled down to it, keeping its direction. 0 disables it.
  double shaping_force_limit_{60.0};
  /// Largest norm, in Nm, of the moment part, scaled the same way. 0 disables it. While
  /// inertia_shaping is on, on_configure sets both limits to 0 when only one of them is positive.
  double shaping_moment_limit_{12.0};

  /// @brief Builds stiffness and damping matrices from a 6-vector of stiffness values.
  /// @param k Stiffness values for each Cartesian DOF.
  ///
  /// The damping is `D = 2 * zeta * sqrt(k * m)`, which damps an inertia m with ratio zeta:
  /// zeta is damping_ratio_, m is translational_inertia_ for the force block and
  /// rotational_inertia_ for the moment block. All three can be set while the controller runs
  /// and take effect through the gains buffer.
  CartesianGains buildGains(const std::array<double, num_cartesian_dof>& k) const;
  /// @brief Friction torque fed forward per FrictionSettings.
  /// @param target_joint_velocity Joint velocity the moving target asks for.
  Eigen::Matrix<double, num_joints, 1> frictionFeedforward(
      const Eigen::Matrix<double, num_joints, 1>& target_joint_velocity);
  /// @brief Moves the reference the target filter follows: to the target, or one cycle along
  /// the segment toward it when the target is interpolated.
  /// @param target The latest received target.
  /// @param dt Length of the cycle in seconds.
  void advanceReference(const TargetPose& target, double dt);
  /// @brief Whether a joint_torque_offset value has 7 finite entries within the limit.
  static bool validJointTorqueOffset(const std::vector<double>& offset);
  /// @brief Whether the friction fractions are non-negative and sum to at most 1, and both
  /// velocity scales are positive.
  static bool validFrictionSettings(const FrictionSettings& settings);
  /// @brief Advances the integral accumulator by one cycle and returns its wrench.
  /// @param error Pose error as computeError returns it.
  /// @param velocity Tool twist in the base frame.
  /// @param dt Length of the cycle in seconds.
  Eigen::Matrix<double, num_cartesian_dof, 1> integralWrench(
      const Eigen::Matrix<double, num_cartesian_dof, 1>& error,
      const Eigen::Matrix<double, num_cartesian_dof, 1>& velocity,
      double dt);
};

}  // namespace franka_example_controllers