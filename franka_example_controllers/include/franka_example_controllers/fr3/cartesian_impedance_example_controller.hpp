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
#include <memory>
#include <mutex>
#include <string>
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
  /// damping_ratio, translational_inertia and rotational_inertia.
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

  /// @brief Builds stiffness and damping matrices from a 6-vector of stiffness values.
  /// @param k Stiffness values for each Cartesian DOF.
  /// Damping design. The vendor formula is `D = 2*sqrt(k)`, which is critical damping for a UNIT
  /// mass; a real arm presents several kilograms of effective Cartesian inertia, so it runs at a
  /// damping ratio near 0.4 -- underdamped at EVERY stiffness, since both the applied and the
  /// required damping scale with sqrt(k). Observed as violent oscillation with nothing touching
  /// the arm, worst with the arm extended sideways where the effective inertia is largest.
  ///
  /// `D = 2 * zeta * sqrt(k * m)` with zeta and m configurable per block. The defaults give
  /// rotation the vendor's value; a rotational inertia below 1 gives it less. All three can be
  /// set while the controller runs and take effect through the gains buffer.
  /// Rate, in 1/s, at which the nullspace target follows the arm. `q_d_nullspace_` is set
  /// once in on_activate and never updated, so moving the tool across the bench builds a
  /// joint error the controller pulls against -- and the nullspace projector is built from a
  /// DAMPED pseudo-inverse, so a fraction of that pull leaks straight into the tool. It lands
  /// mostly as orientation, because rotational stiffness is the weakest axis. Measured on
  /// franko: 4.8 degrees of orientation error at a 98 mm excursion, 10.1 at 485 mm, growing
  /// with distance because the joint error does.
  ///
  /// Letting the target follow keeps that error near zero at any position, so the leak has
  /// nothing to scale with. 0.0 pins it as before, which is the A/B.
  double nullspace_follow_rate_{1.0};
  /// Written by the parameter callback and read by update() when inertia_shaping is on.
  std::atomic<double> damping_ratio_{1.0};
  std::atomic<double> translational_inertia_{6.0};
  std::atomic<double> rotational_inertia_{1.0};
  /// The stiffness last asked for, which a change of the damping design is rebuilt against.
  /// Guarded by gains_mutex_, as is every write to the gains buffer that depends on it.
  std::array<double, num_cartesian_dof> stiffness_target_{};
  std::mutex gains_mutex_;

  /// Per-joint Coulomb friction the arm is believed to have, in Nm, fed forward against the
  /// direction of motion. The impedance law has no integral action, so a static friction torque
  /// stands as a permanent Cartesian offset that no gain change removes.
  std::array<double, num_joints> friction_torque_{};
  /// Fraction of `friction_torque_` to apply. Deliberately under unity: compensating a friction
  /// over-estimate is negative damping at low speed, which is a limit cycle, not a stiffer arm.
  double friction_compensation_{0.0};
  /// Velocity scale, in rad/s, of the tanh that replaces sign(dq). A discrete controller
  /// crossing a discontinuity at 1 kHz chatters; the taper costs a little compensation at a
  /// standstill and removes that.
  double friction_velocity_eps_{0.01};

  /// Integral action on the Cartesian error, N/m per second. The pure impedance law is a
  /// spring: any load it cannot overcome stands as a permanent offset of F/K. Nonzero gain is
  /// the only term here that can remove one.
  double integral_gain_{0.0};
  /// Largest wrench the integrator may contribute, in N and Nm.
  double integral_limit_{30.0};
  /// Rate, in 1/s, at which the accumulator decays while an error is held. The task's long
  /// stationary waits would otherwise charge it against contact and release it as a lurch.
  double integral_leak_{0.5};
  Eigen::Matrix<double, num_cartesian_dof, 1> integral_error_{
      Eigen::Matrix<double, num_cartesian_dof, 1>::Zero()};

  /// How far to cancel the arm's own task-space inertia, 0 for not at all and 1 for
  /// full. `J^T(-K e - D v)` commands a force, so the acceleration it produces divides by
  /// whatever the task inertia is in that posture -- which is why the same commanded step is
  /// brisk near the base and sluggish at reach.
  double inertia_shaping_{0.0};
  /// Project the posture torque with `I - J^T Lambda J M^-1` instead of the damped inverse.
  /// A projector built from a DAMPED pseudo-inverse is not a projector: the damping is
  /// exactly what lets posture effort reach the tool.
  bool consistent_nullspace_{false};
  /// Damping in the inversion of `J M^-1 J^T` for the PROJECTOR. Every unit of it is leak the
  /// projector exists to remove, so it is kept as small as conditioning allows.
  double task_inertia_damping_{0.01};
  /// Damping in the same inversion for the SHAPING term, which needs the opposite of what the
  /// projector needs: it is the only bound on Lambda, which grows without limit as the arm
  /// nears a singularity and multiplies the commanded wrench with it.
  double shaping_damping_{0.2};
  /// Largest shaped force, in N, SCALED to rather than clipped at; 0 disables it. Clipping
  /// componentwise projects onto a box, which moves the wrench's direction: the shaped
  /// components partly cancel through J^T, and breaking that cancellation raises the joint
  /// torque the limit exists to hold down, the more the tighter the box.
  double shaping_force_limit_{60.0};
  /// Largest shaped moment, in Nm, scaled the same way. Separate from the force because they
  /// are not comparable quantities and the wrist -- rated 12 Nm against the shoulder's 87 -- is
  /// the axis this arm is weak on; a single limit loose enough for the force never binds here.
  double shaping_moment_limit_{12.0};

  CartesianGains buildGains(const std::array<double, num_cartesian_dof>& k) const;
  Eigen::Matrix<double, num_joints, 1> frictionFeedforward() const;
  Eigen::Matrix<double, num_cartesian_dof, 1> integralWrench(
      const Eigen::Matrix<double, num_cartesian_dof, 1>& error,
      double dt);
};

}  // namespace franka_example_controllers