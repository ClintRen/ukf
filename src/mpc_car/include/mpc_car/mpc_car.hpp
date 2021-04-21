#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include <arc_spline/arc_spline.hpp>
#include <iosqp/iosqp.hpp>

namespace mpc_car {

static constexpr int n = 4;  // state x y phi v
static constexpr int m = 2;  // input a delta
typedef Eigen::Matrix<double, n, n> MatrixA;
typedef Eigen::Matrix<double, n, m> MatrixB;
typedef Eigen::Vector4d VectorG;
typedef Eigen::Vector4d VectorX;
typedef Eigen::Vector2d VectorU;

class MpcCar {
 private:
  ros::NodeHandle nh_;
  ros::Publisher ref_pub_, traj_pub_;

  double ll_;
  double dt_;
  double rho_;
  int N_;

  double v_max_, a_max_, delta_max_;

  arc_spline::ArcSpline s_;
  double desired_v_;

  osqp::IOSQP qpSolver_;
  Eigen::MatrixXd solMat_, predictMat_;
  std::vector<VectorX> predictState_;
  std::vector<VectorU> predictInput_;

  MatrixA Ad_;
  MatrixB Bd_;
  VectorG gd_;

  Eigen::SparseMatrix<double> P_, q_, A_, l_, u_;
  // Eigen::SparseMatrix<double> P0_, q0_;
  Eigen::SparseMatrix<double> Cx_, lx_, ux_;  // p, v constrains
  Eigen::SparseMatrix<double> Cu_, lu_, uu_;  // a delta vs constrains
  Eigen::SparseMatrix<double> Qx_;

  void linearization(const double& phi,
                     const double& v,
                     const double& delta) {
    Ad_(0, 2) = -v * sin(phi) * dt_;
    Ad_(0, 3) = cos(phi) * dt_;
    Ad_(1, 2) = v * cos(phi) * dt_;
    Ad_(1, 3) = sin(phi) * dt_;
    Ad_(2, 3) = tan(delta) / ll_ * dt_;
    Bd_(2, 1) = v / ll_ / cos(delta) / cos(delta) * dt_;
    gd_(0) = v * sin(phi) * dt_ * phi;
    gd_(1) = -v * cos(phi) * dt_ * phi;
    gd_(2) = -v / ll_ / cos(delta) / cos(delta) * dt_ * delta;
    return;
  }

  void calLinPoint(const double& s0, double& phi, double& v, double& delta) {
    Eigen::Vector2d dxy = s_(s0, 1);
    Eigen::Vector2d ddxy = s_(s0, 2);
    double dx = dxy.x();
    double dy = dxy.y();
    double ddx = ddxy.x();
    double ddy = ddxy.y();
    double dphi = (ddy * dx - dy * ddx) / (dx * dx + dy * dy);
    phi = atan2(dy, dx);
    v = desired_v_;
    delta = atan2(ll_ * dphi, 1.0);
  }

 public:
  MpcCar(ros::NodeHandle& nh) : nh_(nh) {
    // load map
    std::vector<double> track_points_x, track_points_y;
    nh.getParam("track_points_x", track_points_x);
    nh.getParam("track_points_y", track_points_y);
    nh.getParam("desired_v", desired_v_);
    s_.setWayPoints(track_points_x, track_points_y);
    // load parameters
    nh.getParam("ll", ll_);
    nh.getParam("dt", dt_);
    nh.getParam("rho", rho_);
    nh.getParam("N", N_);
    nh.getParam("v_max", v_max_);
    nh.getParam("a_max", a_max_);
    nh.getParam("delta_max", delta_max_);

    ref_pub_ = nh.advertise<nav_msgs::Path>("reference_path", 1);
    traj_pub_ = nh.advertise<nav_msgs::Path>("traj", 1);

    // set Ad, Bd, gd
    Ad_.setIdentity();
    Bd_.setZero();
    Bd_(3, 0) = dt_;
    gd_.setZero();
    // set size of sparse matrices
    P_.resize(m * N_, m * N_);
    q_.resize(m * N_, 1);
    Qx_.resize(n * N_, n * N_);
    Qx_.setIdentity();
    for (int i = 1; i <= N_; ++i) {
      Qx_.coeffRef(i * n - 2, i * n - 2) = rho_;
      Qx_.coeffRef(i * n - 1, i * n - 1) = 0;
    }
    int n_cons = 3;  // v a delta
    A_.resize(n_cons * N_, m * N_);
    l_.resize(n_cons * N_, 1);
    u_.resize(n_cons * N_, 1);
    // v constrains
    Cx_.resize(1 * N_, n * N_);
    lx_.resize(1 * N_, 1);
    ux_.resize(1 * N_, 1);
    // a delta constrains
    Cu_.resize(m * N_, m * N_);
    Cu_.setIdentity();
    lu_.resize(m * N_, 1);
    uu_.resize(m * N_, 1);
    // set lower and upper boundaries
    for (int i = 0; i < N_; ++i) {
      lu_.coeffRef(i * m + 0, 0) = -a_max_;
      uu_.coeffRef(i * m + 0, 0) = a_max_;
      lu_.coeffRef(i * m + 1, 0) = -delta_max_;
      uu_.coeffRef(i * m + 1, 0) = delta_max_;
      Cx_.coeffRef(i, i * n + 3) = 1;
      lx_.coeffRef(i, 0) = -0.1;
      ux_.coeffRef(i, 0) = v_max_;
    }
    // set predict mats size
    predictState_.resize(N_);
    predictInput_.resize(N_);
  }

  int solveQP(const VectorX& x0, VectorU& u) {
    // set BB, AA, gg
    Eigen::MatrixXd BB, AA, gg;
    BB.setZero(n * N_, m * N_);
    AA.setZero(n * N_, n);
    gg.setZero(n * N_, 1);
    double s0 = s_.findS(x0.head(2));
    double phi, v, delta;
    double last_phi = x0(2);
    Eigen::SparseMatrix<double> qx;
    qx.resize(n * N_, 1);
    for (int i = 0; i < N_; ++i) {
      calLinPoint(s0, phi, v, delta);
      if (phi - last_phi > M_PI) {
        phi -= 2 * M_PI;
      } else if (phi - last_phi < -M_PI) {
        phi += 2 * M_PI;
      }
      last_phi = phi;
      linearization(phi, v, delta);
      if (i == 0) {
        BB.block(0, 0, n, m) = Bd_;
        AA.block(0, 0, n, n) = Ad_;
        gg.block(0, 0, n, 1) = gd_;
      } else {
        BB.block(n * i, 0, n, m * N_) = Ad_ * BB.block(n * (i - 1), 0, n, m * N_);
        BB.block(n * i, m * i, n, m) = Bd_;
        AA.block(n * i, 0, n, n) = Ad_ * AA.block(n * (i - 1), 0, n, n);
        gg.block(n * i, 0, n, 1) = Ad_ * gg.block(n * (i - 1), 0, n, 1) + gd_;
      }
      // set qx
      Eigen::Vector2d xy = s_(s0);
      qx.coeffRef(i * n + 0, 0) = -xy.x();
      qx.coeffRef(i * n + 1, 0) = -xy.y();
      qx.coeffRef(i * n + 2, 0) = -rho_ * phi;
      s0 += desired_v_ * dt_;
      s0 = s0 < s_.arcL() ? s0 : s_.arcL();
    }
    Eigen::SparseMatrix<double> BB_sparse = BB.sparseView();
    Eigen::SparseMatrix<double> AA_sparse = AA.sparseView();
    Eigen::SparseMatrix<double> gg_sparse = gg.sparseView();
    Eigen::SparseMatrix<double> x0_sparse = x0.sparseView();

    Eigen::SparseMatrix<double> Cx = Cx_ * BB_sparse;
    Eigen::SparseMatrix<double> lx = lx_ - Cx_ * AA_sparse * x0_sparse - Cx_ * gg_sparse;
    Eigen::SparseMatrix<double> ux = ux_ - Cx_ * AA_sparse * x0_sparse - Cx_ * gg_sparse;
    Eigen::SparseMatrix<double> A_T = A_.transpose();
    A_T.middleCols(0, Cx.rows()) = Cx.transpose();
    A_T.middleCols(Cx.rows(), Cu_.rows()) = Cu_.transpose();
    A_ = A_T.transpose();
    for (int i = 0; i < lx.rows(); ++i) {
      l_.coeffRef(i, 0) = lx.coeff(i, 0);
      u_.coeffRef(i, 0) = ux.coeff(i, 0);
    }
    for (int i = 0; i < lu_.rows(); ++i) {
      l_.coeffRef(i + lx.rows(), 0) = lu_.coeff(i, 0);
      u_.coeffRef(i + lx.rows(), 0) = uu_.coeff(i, 0);
    }
    Eigen::SparseMatrix<double> BBT_sparse = BB_sparse.transpose();
    P_ = BBT_sparse * Qx_ * BB_sparse;
    q_ = BBT_sparse * Qx_.transpose() * (AA_sparse * x0_sparse + gg_sparse) + BBT_sparse * qx;
    // osqp
    Eigen::VectorXd q_d = q_.toDense();
    Eigen::VectorXd l_d = l_.toDense();
    Eigen::VectorXd u_d = u_.toDense();
    qpSolver_.setMats(P_, q_d, A_, l_d, u_d);
    int ret = qpSolver_.solve();
    if (ret != 0) {
      ROS_ERROR("fail to solve QP!");
      return ret;
    }
    Eigen::VectorXd sol = qpSolver_.getPrimalSol();
    solMat_ = Eigen::Map<const Eigen::MatrixXd>(sol.data(), m, N_);
    Eigen::VectorXd solState = BB * sol + AA * x0 + gg;
    predictMat_ = Eigen::Map<const Eigen::MatrixXd>(solState.data(), n, N_);

    u = sol.head(2);
    for (int i = 0; i < N_; ++i) {
      predictInput_[i] = solMat_.col(i);
      predictState_[i] = predictMat_.col(i);
    }
    return ret;
  }

  // visualization
  void visualization() {
    nav_msgs::Path msg;
    msg.header.frame_id = "world";
    msg.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped p;
    for (double s = 0; s < s_.arcL(); s += 0.1) {
      p.pose.position.x = s_(s).x();
      p.pose.position.y = s_(s).y();
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    ref_pub_.publish(msg);
    msg.poses.clear();
    for (int i = 0; i < N_; ++i) {
      p.pose.position.x = predictState_[i](0);
      p.pose.position.y = predictState_[i](1);
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    traj_pub_.publish(msg);
  }
};

}  // namespace mpc_car