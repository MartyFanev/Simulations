#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class Solver {
  public:
    void print_stuff();
    void initialise_input();
    Solver();
    double compute_time_step();
    void simulate();
    MatrixXd euler_flux(MatrixXd v);
    void save_plot();
    void SLIC();
  private:
    double c;
    double a;
    double final_time;
    double x0;
    double x1;
    double dx;
    double dt;
    int cells;
    MatrixXd v;
    MatrixXd v_new;
    MatrixXd flux_initial;
    MatrixXd half_time_step;
    MatrixXd epsalon;
    MatrixXd r;
    MatrixXd v_R_half;
    MatrixXd v_L_half;
    MatrixXd delta;
    MatrixXd delta_downwind;
    MatrixXd delta_upwind;
    MatrixXd rychtmer;
    MatrixXd v_R;
    MatrixXd v_L;
    MatrixXd llf;
    MatrixXd flux;
    double gamma;
};