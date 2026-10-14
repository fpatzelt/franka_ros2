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

#include "franka_example_controllers/fr3/cartesian_impedance_example_controller.hpp"

#include <algorithm>
#include <cassert>

namespace franka_example_controllers {

CartesianImpedanceExampleController::CartesianGains CartesianImpedanceExampleController::buildGains(
    const std::array<double, num_cartesian_dof>& k) const {
  CartesianGains g;
  for (int i = 0; i < num_cartesian_dof; ++i) {
    const double ki = std::max(0.0, k[i]);
    g.stiffness(i, i) = ki;

    // 2 * zeta * sqrt(k * m) damps an inertia m with ratio zeta. The vendor's 2 * sqrt(k) is the
    // case zeta = 1, m = 1.
    const double inertia = (i < 3) ? translational_inertia_.load() : rotational_inertia_.load();
    g.damping(i, i) = 2.0 * damping_ratio_.load() * std::sqrt(ki * inertia);
  }
  return g;
}

CartesianImpedanceExampleController::CallbackReturn CartesianImpedanceExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<double>("nullspace_stiffness", 20.0);
    auto_declare<double>("filter_params", 0.005);
    auto_declare<double>("translational_stiffness", 150.0);
    auto_declare<double>("rotational_stiffness", 10.0);
    auto_declare<double>("nullspace_follow_rate", 1.0);
    auto_declare<double>("damping_ratio", 1.0);
    auto_declare<double>("translational_inertia", 3.0);
    auto_declare<double>("rotational_inertia", 0.25);
    auto_declare<double>("velocity_feedforward", 0.0);
    auto_declare<double>("acceleration_feedforward", 0.0);
    auto_declare<bool>("interpolate_target", false);
    auto_declare<std::vector<double>>("joint_torque_offset", std::vector<double>(num_joints, 0.0));
    auto_declare<std::vector<double>>("friction_torque", std::vector<double>(num_joints, 0.0));
    for (const auto& [name, field] : k_friction_parameters_) {
      auto_declare<double>(name, FrictionSettings{}.*field);
    }
    for (const auto& [name, field] : k_integral_parameters_) {
      auto_declare<double>(name, IntegralSettings{}.*field);
    }
    auto_declare<double>("inertia_shaping", 0.0);
    auto_declare<bool>("consistent_nullspace", false);
    auto_declare<double>("task_inertia_damping", 0.01);
    auto_declare<double>("shaping_damping", 0.2);
    auto_declare<double>("shaping_force_limit", 60.0);
    auto_declare<double>("shaping_moment_limit", 12.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; i++) {
    command_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return command_interfaces_config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // 1) Joint position/velocity — read directly from hardware_interface state interfaces.
  //    Order is (j1.pos, j1.vel, j2.pos, j2.vel, ...) so readJointState() can index by 2*i.
  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }

  // 2) Cartesian pose state via FrankaCartesianPoseInterface (16 column-major matrix entries).
  for (const auto& name : franka_cartesian_pose_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(name);
  }

  // 3) Robot model + state for mass, coriolis, jacobian.
  for (const auto& name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(name);
  }

  return state_interfaces_config;
}

CartesianImpedanceExampleController::CallbackReturn
CartesianImpedanceExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  // Match the franka_hardware export pattern: prefix + index + "/cartesian_pose_state".
  const std::string cartesian_pose_prefix = arm_prefix_.empty() ? std::string{} : arm_prefix_ + "_";

  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          cartesian_pose_prefix, k_elbow_activated_);

  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id_ + "/" + k_robot_model_interface_name, arm_id_ + "/" + k_robot_state_interface_name);

  // A zero would leave an axis undamped and a negative one makes the square root a NaN, so the
  // controller refuses to configure on either.
  for (const char* name : {"damping_ratio", "translational_inertia", "rotational_inertia"}) {
    const double value = get_node()->get_parameter(name).as_double();
    if (!std::isfinite(value) || value <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s must be finite and positive, got %f", name, value);
      return CallbackReturn::ERROR;
    }
  }

  for (const auto& [name, field] : k_integral_parameters_) {
    const double value = get_node()->get_parameter(name).as_double();
    if (!std::isfinite(value) || value < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s must be finite and non-negative, got %f", name,
                   value);
      return CallbackReturn::ERROR;
    }
    integral_settings_.*field = value;
  }
  integral_buffer_.initRT(integral_settings_);

  const double feedforward = get_node()->get_parameter("velocity_feedforward").as_double();
  if (!std::isfinite(feedforward) || feedforward < 0.0 || feedforward > 1.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "velocity_feedforward must be within [0, 1], got %f",
                 feedforward);
    return CallbackReturn::ERROR;
  }
  velocity_feedforward_ = feedforward;

  const double acceleration = get_node()->get_parameter("acceleration_feedforward").as_double();
  if (!std::isfinite(acceleration) || acceleration < 0.0 || acceleration > 1.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "acceleration_feedforward must be within [0, 1], got %f",
                 acceleration);
    return CallbackReturn::ERROR;
  }
  acceleration_feedforward_ = acceleration;
  interpolate_target_ = get_node()->get_parameter("interpolate_target").as_bool();

  const auto offset = get_node()->get_parameter("joint_torque_offset").as_double_array();
  std::array<double, num_joints> joint_torque_offset{};
  if (!validJointTorqueOffset(offset)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "joint_torque_offset needs %d finite values within +-%.1f Nm", num_joints,
                 k_max_joint_torque_offset_);
    return CallbackReturn::ERROR;
  }
  std::copy(offset.begin(), offset.end(), joint_torque_offset.begin());
  joint_torque_offset_buffer_.initRT(joint_torque_offset);

  for (const auto& [name, field] : k_friction_parameters_) {
    friction_settings_.*field = get_node()->get_parameter(name).as_double();
  }
  if (!validFrictionSettings(friction_settings_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "friction settings: fractions must be non-negative and sum to at most 1, "
                 "velocity scales positive");
    return CallbackReturn::ERROR;
  }

  nullspace_follow_rate_ = get_node()->get_parameter("nullspace_follow_rate").as_double();
  damping_ratio_ = get_node()->get_parameter("damping_ratio").as_double();
  translational_inertia_ = get_node()->get_parameter("translational_inertia").as_double();
  rotational_inertia_ = get_node()->get_parameter("rotational_inertia").as_double();
  inertia_shaping_ = get_node()->get_parameter("inertia_shaping").as_double();
  consistent_nullspace_ = get_node()->get_parameter("consistent_nullspace").as_bool();
  task_inertia_damping_ = get_node()->get_parameter("task_inertia_damping").as_double();
  shaping_damping_ = get_node()->get_parameter("shaping_damping").as_double();
  shaping_force_limit_ = get_node()->get_parameter("shaping_force_limit").as_double();
  shaping_moment_limit_ = get_node()->get_parameter("shaping_moment_limit").as_double();

  // Disabling one half asks for less clamping and delivers more torque: J^T w is
  // J_v^T f + J_w^T m and the two partly cancel, so rescaling one half alone breaks the
  // cancellation and raises peak joint torque above no clamp at all. Refused rather than
  // half-applied, the same treatment a wrong-length friction_torque gets.
  if (inertia_shaping_ > 0.0 &&
      ((shaping_force_limit_ > 0.0) != (shaping_moment_limit_ > 0.0))) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "shaping_force_limit %.1f and shaping_moment_limit %.1f: clamping one half "
                 "alone raises peak joint torque above no clamp at all; both disabled",
                 shaping_force_limit_, shaping_moment_limit_);
    shaping_force_limit_ = 0.0;
    shaping_moment_limit_ = 0.0;
  }

  if (acceleration_feedforward_ > 0.0 &&
      !(shaping_force_limit_ > 0.0 && shaping_moment_limit_ > 0.0)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "acceleration_feedforward needs shaping_force_limit and shaping_moment_limit "
                 "both positive");
    return CallbackReturn::ERROR;
  }

  const auto friction = get_node()->get_parameter("friction_torque").as_double_array();
  if (friction.size() == num_joints) {
    std::copy(friction.begin(), friction.end(), friction_torque_.begin());
  } else if (friction_settings_.compensation > 0.0 ||
             friction_settings_.target_compensation > 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "friction_torque needs %d values, got %zu; compensation disabled", num_joints,
                 friction.size());
    friction_settings_.compensation = 0.0;
    friction_settings_.target_compensation = 0.0;
  }
  friction_buffer_.initRT(friction_settings_);

  const double t_k = get_node()->get_parameter("translational_stiffness").as_double();
  const double r_k = get_node()->get_parameter("rotational_stiffness").as_double();
  const double n_k = get_node()->get_parameter("nullspace_stiffness").as_double();

  CartesianGains gains;
  {
    std::lock_guard<std::mutex> lock(gains_mutex_);
    stiffness_target_ = {t_k, t_k, t_k, r_k, r_k, r_k};
    gains = buildGains(stiffness_target_);
  }
  cartesian_stiffness_ = gains.stiffness;
  cartesian_damping_ = gains.damping;
  nullspace_stiffness_ = n_k;
  cartesian_gains_buffer_.initRT(gains);
  nullspace_stiffness_buffer_.initRT(n_k);

  // One minus the pole of the first-order lag the target is filtered through, per control
  // cycle: at 1 kHz, 0.005 is a 200 ms time constant and 0.02 is 50 ms. Larger tracks a
  // streamed target more closely and passes more of its jitter through to the arm.
  filter_params_ = get_node()->get_parameter("filter_params").as_double();
  filter_params_buffer_.initRT(filter_params_);

  target_pose_buffer_.initRT(TargetPose{});

  sub_equilibrium_pose_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/equilibrium_pose", rclcpp::SystemDefaultsQoS(),
      std::bind(&CartesianImpedanceExampleController::equilibriumPoseCallback, this,
                std::placeholders::_1));

  srv_set_cartesian_stiffness_ =
      get_node()->create_service<franka_msgs::srv::SetCartesianStiffness>(
          "~/set_cartesian_stiffness",
          std::bind(&CartesianImpedanceExampleController::setCartesianStiffnessCallback, this,
                    std::placeholders::_1, std::placeholders::_2));

  param_cb_handle_ = get_node()->add_on_set_parameters_callback(std::bind(
      &CartesianImpedanceExampleController::onParameterUpdate, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CartesianImpedanceExampleController::CallbackReturn
CartesianImpedanceExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);

  readJointState();
  Eigen::Quaterniond orientation_init;
  Eigen::Vector3d position_init;
  std::tie(orientation_init, position_init) =
      franka_cartesian_pose_->getCurrentOrientationAndTranslation();
  orientation_init.normalize();

  position_d_ = position_init;
  initial_position_ = position_init;
  elapsed_time_ = 0.0;
  orientation_d_ = orientation_init;
  q_d_nullspace_ = q_;
  integral_wrench_.setZero();
  target_twist_.setZero();
  target_twist_filtered_.setZero();
  target_acceleration_.setZero();
  reference_position_ = segment_start_position_ = segment_end_position_ = position_init;
  reference_orientation_ = segment_start_orientation_ = segment_end_orientation_ = orientation_init;
  segment_elapsed_ = arrival_interval_;
  since_arrival_ = 1.0;
  arrival_interval_measured_ = false;

  TargetPose initial_target;
  initial_target.position = position_init;
  initial_target.orientation = orientation_init;
  initial_target.sequence = target_sequence_.load();
  seen_sequence_ = initial_target.sequence;
  target_pose_buffer_.initRT(initial_target);

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianImpedanceExampleController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();
  franka_cartesian_pose_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void CartesianImpedanceExampleController::readJointState() {
  // The first 2 * num_joints entries are joint position/velocity, in the order declared
  // by state_interface_configuration().
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_optional().value();
    dq_(i) = velocity_interface.get_optional().value();
  }
}

controller_interface::return_type CartesianImpedanceExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  readJointState();

  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;
  std::tie(orientation, position) = franka_cartesian_pose_->getCurrentOrientationAndTranslation();

  std::array<double, num_joints> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, num_cartesian_dof * num_joints> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);

  Eigen::Map<Eigen::Matrix<double, num_joints, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, num_cartesian_dof, num_joints>> jacobian(jacobian_array.data());

  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() = position;
  transform.rotate(orientation.toRotationMatrix());

  const TargetPose target = *target_pose_buffer_.readFromRT();
  //updateMotionTarget(period.seconds(), target.orientation);

  filter_params_ = *filter_params_buffer_.readFromRT();
  const CartesianGains target_gains = *cartesian_gains_buffer_.readFromRT();
  const double target_nullspace_stiffness = *nullspace_stiffness_buffer_.readFromRT();

  const auto error = computeError(position, orientation, transform);

  Eigen::Matrix<double, num_joints, 1> tau_task, tau_nullspace, tau_d;

  // Damped pseudo-inverse of jacobian transpose
  Eigen::MatrixXd jacobian_transpose_pinv;
  double lambda = 0.2;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(),
                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals = svd.singularValues();
  Eigen::MatrixXd S = jacobian.transpose();
  S.setZero();
  for (int i = 0; i < sing_vals.size(); i++) {
    S(i, i) = sing_vals(i) / (sing_vals(i) * sing_vals(i) + lambda * lambda);
  }
  jacobian_transpose_pinv = svd.matrixV() * S.transpose() * svd.matrixU().transpose();

  const Eigen::Matrix<double, num_cartesian_dof, 1> velocity = jacobian * dq_;
  const bool interpolate = interpolate_target_.load();
  const Eigen::Matrix<double, num_cartesian_dof, 1>& feedforward_twist =
      interpolate ? target_twist_ : target_twist_filtered_;
  Eigen::Matrix<double, num_cartesian_dof, 1> wrench =
      -cartesian_stiffness_ * error -
      cartesian_damping_ * (velocity - velocity_feedforward_.load() * feedforward_twist);
  wrench += integralWrench(error, velocity, period.seconds());

  const auto identity_joints = Eigen::Matrix<double, num_joints, num_joints>::Identity();
  const auto identity_task =
      Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof>::Identity();

  // The mass matrix is read only when a term needs it: both users invert a 6x6 built from it,
  // which is not free at 1 kHz and is pure waste under the default configuration.
  Eigen::Matrix<double, num_joints, num_cartesian_dof> mass_inverse_jacobian_transpose =
      Eigen::Matrix<double, num_joints, num_cartesian_dof>::Zero();
  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> task_inertia = identity_task;
  Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> shaping_inertia = identity_task;
  const double acceleration_feedforward = interpolate ? acceleration_feedforward_.load() : 0.0;
  if (inertia_shaping_ > 0.0 || consistent_nullspace_ || acceleration_feedforward > 0.0) {
    std::array<double, num_joints * num_joints> mass_array = franka_robot_model_->getMassMatrix();
    const Eigen::Map<Eigen::Matrix<double, num_joints, num_joints>> mass(mass_array.data());
    mass_inverse_jacobian_transpose = mass.ldlt().solve(jacobian.transpose());
    const Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> task_inverse =
        jacobian * mass_inverse_jacobian_transpose;
    task_inertia =
        (task_inverse + task_inertia_damping_ * task_inertia_damping_ * identity_task).inverse();
    shaping_inertia =
        (task_inverse + shaping_damping_ * shaping_damping_ * identity_task).inverse();
  }

  // Premultiplying by Lambda Lambda_d^-1 cancels the arm's real task inertia and leaves the
  // response the gains describe, identically everywhere in the workspace. Lambda_d is the
  // notional inertia the damping is already built against, so a shaping of 0 is the identity.
  if (inertia_shaping_ > 0.0) {
    Eigen::Matrix<double, num_cartesian_dof, num_cartesian_dof> shape = shaping_inertia;
    for (int i = 0; i < num_cartesian_dof; i++) {
      shape.col(i) /= (i < 3 ? translational_inertia_.load() : rotational_inertia_.load());
    }
    wrench = ((1.0 - inertia_shaping_) * identity_task + inertia_shaping_ * shape) * wrench;
    const double force = wrench.head<3>().norm();
    if (shaping_force_limit_ > 0.0 && force > shaping_force_limit_) {
      wrench.head<3>() *= shaping_force_limit_ / force;
    }
    const double moment = wrench.tail<3>().norm();
    if (shaping_moment_limit_ > 0.0 && moment > shaping_moment_limit_) {
      wrench.tail<3>() *= shaping_moment_limit_ / moment;
    }
  }

  if (acceleration_feedforward > 0.0) {
    Eigen::Matrix<double, num_cartesian_dof, 1> inertial =
        acceleration_feedforward * shaping_inertia * target_acceleration_;
    const double force = inertial.head<3>().norm();
    if (shaping_force_limit_ > 0.0 && force > shaping_force_limit_) {
      inertial.head<3>() *= shaping_force_limit_ / force;
    }
    const double moment = inertial.tail<3>().norm();
    if (shaping_moment_limit_ > 0.0 && moment > shaping_moment_limit_) {
      inertial.tail<3>() *= shaping_moment_limit_ / moment;
    }
    wrench += inertial;
  }

  tau_task << jacobian.transpose() * wrench;

  const Eigen::Matrix<double, num_joints, 1> posture =
      nullspace_stiffness_ * (q_d_nullspace_ - q_) - 2.0 * std::sqrt(nullspace_stiffness_) * dq_;
  Eigen::Matrix<double, num_joints, num_joints> projector;
  if (consistent_nullspace_) {
    projector = identity_joints -
                jacobian.transpose() * task_inertia * mass_inverse_jacobian_transpose.transpose();
  } else {
    projector = identity_joints - jacobian.transpose() * jacobian_transpose_pinv;
  }
  tau_nullspace << projector * posture;

  // Damped least squares, J^T (J J^T + 1e-4 I)^-1, for the joint velocity the target asks for.
  const Eigen::Matrix<double, num_joints, 1> target_joint_velocity =
      jacobian.transpose() *
      (jacobian * jacobian.transpose() + 1e-4 * identity_task).ldlt().solve(feedforward_twist);
  const std::array<double, num_joints> joint_torque_offset =
      *joint_torque_offset_buffer_.readFromRT();

  tau_d << tau_task + tau_nullspace + coriolis + frictionFeedforward(target_joint_velocity) +
               Eigen::Map<const Eigen::Matrix<double, num_joints, 1>>(joint_torque_offset.data());

  for (int i = 0; i < num_joints; i++) {
    if (!command_interfaces_[i].set_value(tau_d[i])) {
      return controller_interface::return_type::ERROR;
    }
  }

  // Let the nullspace target follow the arm, so the joint error it pulls against cannot grow
  // with the excursion. Clamped to [0,1] so a large rate degenerates to "track exactly" rather
  // than overshooting past the measurement and pulling the other way.
  if (nullspace_follow_rate_ > 0.0) {
    const double follow = std::min(1.0, std::max(0.0, nullspace_follow_rate_ * period.seconds()));
    q_d_nullspace_ += follow * (q_ - q_d_nullspace_);
  }

  cartesian_stiffness_ =
      filter_params_ * target_gains.stiffness + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * target_gains.damping + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * target_nullspace_stiffness + (1.0 - filter_params_) * nullspace_stiffness_;

  // The reference and the target filter move once per update, so their rates use the nominal
  // cycle length, or the measured period when no update rate is set.
  const double cycle = get_update_rate() > 0 ? 1.0 / get_update_rate() : period.seconds();
  advanceReference(target, cycle);

  const Eigen::Vector3d previous_position_d = position_d_;
  const Eigen::Quaterniond previous_orientation_d = orientation_d_;
  position_d_ = filter_params_ * reference_position_ + (1.0 - filter_params_) * position_d_;
  // Keep the working orientation on the same hemisphere as the reference before slerp.
  if (orientation_d_.coeffs().dot(reference_orientation_.coeffs()) < 0.0) {
    orientation_d_.coeffs() = -orientation_d_.coeffs();
  }
  orientation_d_ = orientation_d_.slerp(filter_params_, reference_orientation_);
  orientation_d_.normalize();

  if (cycle > 0.0) {
    const Eigen::Matrix<double, num_cartesian_dof, 1> previous_twist = target_twist_;
    target_twist_.head<3>() = (position_d_ - previous_position_d) / cycle;
    const Eigen::AngleAxisd turn(orientation_d_ * previous_orientation_d.inverse());
    target_twist_.tail<3>() = turn.angle() / cycle * turn.axis();
    target_acceleration_ = filter_params_ * (target_twist_ - previous_twist) / cycle +
                           (1.0 - filter_params_) * target_acceleration_;
  }
  target_twist_filtered_ =
      filter_params_ * target_twist_ + (1.0 - filter_params_) * target_twist_filtered_;

  return controller_interface::return_type::OK;
}

void CartesianImpedanceExampleController::updateMotionTarget(
    double period_seconds,
    const Eigen::Quaterniond& orientation) {
  elapsed_time_ += period_seconds;
  double angle = M_PI / 4.0 * (1 - std::cos(M_PI / 5.0 * elapsed_time_));
  double delta_x = 0.1 * std::sin(angle);
  double delta_z = 0.1 * (std::cos(angle) - 1);
  TargetPose motion_target;
  motion_target.position = initial_position_ + Eigen::Vector3d(delta_x, 0.0, delta_z);
  motion_target.orientation = orientation;
  target_pose_buffer_.writeFromNonRT(motion_target);
}

Eigen::Matrix<double, 6, 1> CartesianImpedanceExampleController::computeError(
    const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation,
    const Eigen::Affine3d& transform) const {
  Eigen::Matrix<double, num_cartesian_dof, 1> error;
  error.head(3) << position - position_d_;
  Eigen::Quaterniond orientation_corrected = orientation;
  if (orientation_d_.coeffs().dot(orientation_corrected.coeffs()) < 0.0) {
    orientation_corrected.coeffs() = -orientation_corrected.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation_corrected.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);
  return error;
}

void CartesianImpedanceExampleController::equilibriumPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  TargetPose target;
  target.position << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  target.orientation = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                          msg->pose.orientation.y, msg->pose.orientation.z);

  // Reject malformed input rather than letting NaNs poison the RT thread.
  if (target.orientation.coeffs().norm() < 1e-6) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Discarding equilibrium pose with degenerate quaternion.");
    return;
  }
  target.orientation.normalize();
  target.sequence = ++target_sequence_;

  target_pose_buffer_.writeFromNonRT(target);
}

void CartesianImpedanceExampleController::setCartesianStiffnessCallback(
    const std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Request> request,
    std::shared_ptr<franka_msgs::srv::SetCartesianStiffness::Response> response) {
  std::array<double, num_cartesian_dof> k{};
  bool valid = true;
  for (size_t i = 0; i < num_cartesian_dof; ++i) {
    k[i] = request->cartesian_stiffness[i];
    if (!std::isfinite(k[i]) || k[i] < 0.0) {
      valid = false;
      break;
    }
  }
  if (!valid) {
    response->success = false;
    response->error = "Cartesian stiffness must contain 6 finite, non-negative values.";
    return;
  }

  {
    std::lock_guard<std::mutex> lock(gains_mutex_);
    stiffness_target_ = k;
    cartesian_gains_buffer_.writeFromNonRT(buildGains(k));
  }
  response->success = true;
  response->error = "";
}

rcl_interfaces::msg::SetParametersResult CartesianImpedanceExampleController::onParameterUpdate(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  const auto designs_damping = [](const std::string& name) {
    return name == "damping_ratio" || name == "translational_inertia" ||
           name == "rotational_inertia";
  };
  const auto friction_field = [](const std::string& name) -> double FrictionSettings::* {
    for (const auto& [parameter, field] : k_friction_parameters_) {
      if (name == parameter) {
        return field;
      }
    }
    return nullptr;
  };
  FrictionSettings friction = friction_settings_;
  bool friction_changed = false;
  const auto integral_field = [](const std::string& name) -> double IntegralSettings::* {
    for (const auto& [parameter, field] : k_integral_parameters_) {
      if (name == parameter) {
        return field;
      }
    }
    return nullptr;
  };

  // Every value is checked before any is applied, so a rejected request changes nothing.
  for (const auto& p : parameters) {
    const std::string& name = p.get_name();
    if (name == "nullspace_stiffness") {
      const double v = p.as_double();
      if (!std::isfinite(v) || v < 0.0) {
        result.successful = false;
        result.reason = "nullspace_stiffness must be finite and non-negative";
        return result;
      }
    } else if (name == "filter_params") {
      const double v = p.as_double();
      if (!std::isfinite(v) || v <= 0.0 || v > 1.0) {
        result.successful = false;
        result.reason = "filter_params must be finite and within (0, 1]";
        return result;
      }
    } else if (designs_damping(name)) {
      const double v = p.as_double();
      if (!std::isfinite(v) || v <= 0.0) {
        result.successful = false;
        result.reason = name + " must be finite and positive";
        return result;
      }
    } else if (name == "velocity_feedforward" || name == "acceleration_feedforward") {
      const double v = p.as_double();
      if (!std::isfinite(v) || v < 0.0 || v > 1.0) {
        result.successful = false;
        result.reason = name + " must be within [0, 1]";
        return result;
      }
      if (name == "acceleration_feedforward" && v > 0.0 &&
          !(shaping_force_limit_ > 0.0 && shaping_moment_limit_ > 0.0)) {
        result.successful = false;
        result.reason =
            "acceleration_feedforward needs shaping_force_limit and shaping_moment_limit positive";
        return result;
      }
    } else if (name == "joint_torque_offset") {
      if (!validJointTorqueOffset(p.as_double_array())) {
        result.successful = false;
        result.reason = "joint_torque_offset needs 7 finite values of at most 2 Nm";
        return result;
      }
    } else if (const auto field = friction_field(name)) {
      friction.*field = p.as_double();
      friction_changed = true;
    } else if (integral_field(name) != nullptr) {
      const double v = p.as_double();
      if (!std::isfinite(v) || v < 0.0) {
        result.successful = false;
        result.reason = name + " must be finite and non-negative";
        return result;
      }
    }
  }

  if (friction_changed && !validFrictionSettings(friction)) {
    result.successful = false;
    result.reason =
        "friction fractions must be non-negative and sum to at most 1, velocity scales positive";
    return result;
  }

  bool damping_changed = false;
  bool integral_changed = false;
  for (const auto& p : parameters) {
    const std::string& name = p.get_name();
    if (name == "nullspace_stiffness") {
      nullspace_stiffness_buffer_.writeFromNonRT(p.as_double());
    } else if (name == "filter_params") {
      filter_params_buffer_.writeFromNonRT(p.as_double());
    } else if (name == "damping_ratio") {
      damping_ratio_ = p.as_double();
      damping_changed = true;
    } else if (name == "translational_inertia") {
      translational_inertia_ = p.as_double();
      damping_changed = true;
    } else if (name == "rotational_inertia") {
      rotational_inertia_ = p.as_double();
      damping_changed = true;
    } else if (name == "velocity_feedforward") {
      velocity_feedforward_ = p.as_double();
    } else if (name == "acceleration_feedforward") {
      acceleration_feedforward_ = p.as_double();
    } else if (name == "interpolate_target") {
      interpolate_target_ = p.as_bool();
    } else if (name == "joint_torque_offset") {
      const auto values = p.as_double_array();
      std::array<double, num_joints> offset{};
      std::copy(values.begin(), values.end(), offset.begin());
      joint_torque_offset_buffer_.writeFromNonRT(offset);
    } else if (const auto field = integral_field(name)) {
      integral_settings_.*field = p.as_double();
      integral_changed = true;
    }
  }
  if (integral_changed) {
    integral_buffer_.writeFromNonRT(integral_settings_);
  }
  if (friction_changed) {
    friction_settings_ = friction;
    friction_buffer_.writeFromNonRT(friction_settings_);
  }
  // The damping is rebuilt for the stiffness in effect and reaches update() through the gains
  // buffer, which blends it in at the rate of filter_params like a new stiffness.
  if (damping_changed) {
    std::lock_guard<std::mutex> lock(gains_mutex_);
    cartesian_gains_buffer_.writeFromNonRT(buildGains(stiffness_target_));
  }
  return result;
}

Eigen::Matrix<double, CartesianImpedanceExampleController::num_joints, 1>
CartesianImpedanceExampleController::frictionFeedforward(
    const Eigen::Matrix<double, num_joints, 1>& target_joint_velocity) {
  const FrictionSettings settings = *friction_buffer_.readFromRT();
  Eigen::Matrix<double, num_joints, 1> tau_friction = Eigen::Matrix<double, num_joints, 1>::Zero();
  for (int i = 0; i < num_joints; i++) {
    tau_friction[i] = friction_torque_[i] *
                      (settings.compensation * std::tanh(dq_[i] / settings.velocity_eps) +
                       settings.target_compensation *
                           std::tanh(target_joint_velocity[i] / settings.target_velocity_eps));
  }
  return tau_friction;
}

void CartesianImpedanceExampleController::advanceReference(const TargetPose& target, double dt) {
  since_arrival_ += dt;
  const bool arrived = target.sequence != seen_sequence_;
  seen_sequence_ = target.sequence;
  if (!interpolate_target_.load()) {
    reference_position_ = segment_start_position_ = segment_end_position_ = target.position;
    reference_orientation_ = segment_start_orientation_ = segment_end_orientation_ =
        target.orientation;
    segment_elapsed_ = arrival_interval_;
    return;
  }
  if (arrived) {
    if (since_arrival_ >= k_min_arrival_interval_ && since_arrival_ <= k_max_arrival_interval_) {
      arrival_interval_ =
          arrival_interval_measured_
              ? arrival_interval_ + k_arrival_smoothing_ * (since_arrival_ - arrival_interval_)
              : since_arrival_;
      arrival_interval_measured_ = true;
    }
    since_arrival_ = 0.0;
    segment_start_position_ = reference_position_;
    segment_start_orientation_ = reference_orientation_;
    segment_end_position_ = target.position;
    segment_end_orientation_ = target.orientation;
    segment_elapsed_ = 0.0;
  }
  segment_elapsed_ = std::min(segment_elapsed_ + dt, arrival_interval_);
  const double fraction = segment_elapsed_ / arrival_interval_;
  reference_position_ =
      segment_start_position_ + fraction * (segment_end_position_ - segment_start_position_);
  reference_orientation_ = segment_start_orientation_.slerp(fraction, segment_end_orientation_);
}

bool CartesianImpedanceExampleController::validJointTorqueOffset(
    const std::vector<double>& offset) {
  return offset.size() == num_joints && std::all_of(offset.begin(), offset.end(), [](double v) {
           return std::isfinite(v) && std::abs(v) <= k_max_joint_torque_offset_;
         });
}

bool CartesianImpedanceExampleController::validFrictionSettings(const FrictionSettings& settings) {
  return std::isfinite(settings.compensation) && std::isfinite(settings.target_compensation) &&
         settings.compensation >= 0.0 && settings.target_compensation >= 0.0 &&
         settings.compensation + settings.target_compensation <= 1.0 &&
         std::isfinite(settings.velocity_eps) && settings.velocity_eps > 0.0 &&
         std::isfinite(settings.target_velocity_eps) && settings.target_velocity_eps > 0.0;
}

Eigen::Matrix<double, CartesianImpedanceExampleController::num_cartesian_dof, 1>
CartesianImpedanceExampleController::integralWrench(
    const Eigen::Matrix<double, num_cartesian_dof, 1>& error,
    const Eigen::Matrix<double, num_cartesian_dof, 1>& velocity,
    double dt) {
  const IntegralSettings settings = *integral_buffer_.readFromRT();
  if (settings.rate <= 0.0) {
    integral_wrench_.setZero();
    return integral_wrench_;
  }

  const bool settled = target_twist_.head<3>().norm() < settings.max_speed &&
                       target_twist_.tail<3>().norm() < settings.max_angular_speed &&
                       velocity.head<3>().norm() < settings.max_speed &&
                       velocity.tail<3>().norm() < settings.max_angular_speed;
  if (settled) {
    // The part of a vector that reaches beyond a ball of the given radius.
    const auto beyond = [](const Eigen::Vector3d& v, double radius) -> Eigen::Vector3d {
      const double norm = v.norm();
      return norm > radius ? Eigen::Vector3d((1.0 - radius / norm) * v) : Eigen::Vector3d::Zero();
    };
    // The orientation error is the error quaternion's vector part, sin(angle / 2) long.
    Eigen::Matrix<double, num_cartesian_dof, 1> outside;
    outside << beyond(error.head<3>(), settings.position_deadband),
        beyond(error.tail<3>(), std::sin(0.5 * settings.orientation_deadband));
    integral_wrench_ -= settings.rate * dt * (cartesian_stiffness_ * outside);
  } else {
    integral_wrench_ *= std::max(0.0, 1.0 - settings.rate * dt);
  }

  const double force = integral_wrench_.head<3>().norm();
  if (force > settings.force_limit) {
    integral_wrench_.head<3>() *= settings.force_limit / force;
  }
  const double moment = integral_wrench_.tail<3>().norm();
  if (moment > settings.moment_limit) {
    integral_wrench_.tail<3>() *= settings.moment_limit / moment;
  }
  return integral_wrench_;
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerInterface)