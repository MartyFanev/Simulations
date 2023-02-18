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
    double minbee(double r);
    double Van_Albada(double r);
    MatrixXd compute_cf(MatrixXd v);


  private:
    double c;
    double a;
    double final_time;
    double x0;
    double x1;
    double dx;
    double dt;
    int cells;
    double p_inf;
    double gamma;
    double Bx; // in 1D MHD Bx is always constant
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
    MatrixXd v_R_flux;
    MatrixXd v_L_flux;
    MatrixXd llf;
    MatrixXd flux;
    MatrixXd hllc;
    MatrixXd star_l;
    MatrixXd star_r;
};

Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.9;
  cells = 400;
  a = 1.;
  final_time = 80.;
  x0 = 0.;
  x1 = 800.;
  dx = (x1-x0)/cells;
  gamma = 2.; // use gamma for stiffened water
  p_inf = 6e8;
  v.resize(7,cells+2); // set up the vector for the solution
  v_L_flux.resize(7,v.cols());
  v_R_flux.resize(7,v.cols());
  v_new.resize(7,cells+2); // set up the placement vector for the solutions
  epsalon.resize(7,v.cols());
  half_time_step.resize(7,v.cols());
  v_L_half.resize(7,v.cols());
  v_R_half.resize(7,v.cols());
  r.resize(7,v.cols());
  v_L.resize(7,v.cols());
  v_R.resize(7,v.cols());
  delta_upwind.resize(7,v.cols());
  delta_downwind.resize(7,v.cols());
  delta.resize(7,v.cols());
  flux_initial.resize(7,cells+2);
  flux.resize(7,cells+2);
  hllc.resize(7,v.cols());
  star_l.resize(7,v.cols());
  star_r.resize(7,v.cols());
  Bx=0.75;
}

void Solver::initialise_input(){
  std::ofstream input("input.dat");

// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < v.cols(); i++){
    double x = x0 + (i-0.5)*dx;

    if(x < 400.0){

      
      double density = 1.0;
      double velocity_x = 0.;
      double velocity_y = 0.;
      double velocity_z =0.;
      double By=1.;
      double Bz=0.;
      double pressure=1.;
      double Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
      double energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      v(0,i) = density;
      v(1,i) = velocity_x * density; // the momentum value y*density
      v(2,i) = velocity_y*density;
      v(3,i) = velocity_z*density;
      v(4,i) = energy;
      v(5,i) = By;
      v(6,i) = Bz;

    }else if (x>400.0){

      double density = .125;
      double velocity_x = 0.;
      double velocity_y = 0.;
      double velocity_z =0.;
      double By=-1.;
      double Bz=0.;
      double pressure=0.1; // this variable will be a standalone for total_pressure
      double Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
      double energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      v(0,i) = density;
      v(1,i) = velocity_x * density; // the momentum value y*density
      v(2,i) = velocity_y*density;
      v(3,i) = velocity_z*density;
      v(4,i) = energy;
      v(5,i) = By;
      v(6,i) = Bz;

    }
    
    input << x << " " << v(0,i) << " " << v(1,i)/v(0,i) << " " <<  (v(2,i)-(0.5*v(0,i)*pow((v(1,i)/v(0,i)),2)))*(gamma-1) << "\n";
  
  }
}


double Solver::compute_time_step() {

// Compute the time step for each iteration

  VectorXd a(v.cols());
  MatrixXd abs_v = v.cwiseAbs();


  for (int i=0 ; i < abs_v.cols(); i++){

    double velocity_x = abs_v(1,i)/abs_v(0,i);
    double velocity_y = abs_v(2,i)/abs_v(0,i);
    double velocity_z = abs_v(3,i)/abs_v(0,i);
    double total_velocity = pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2);
    double Bsqrd = pow(v(6,i),2)+pow(v(5,i),2)+pow(Bx,2); 
    double pressure = ((v(4,i)-(0.5*v(0,i)*total_velocity) - 0.5*Bsqrd)*(gamma-1));

    double cs = std::sqrt((gamma * (pressure)) /v(0,i));
    double ca  =std::sqrt(Bsqrd)/std::sqrt(v(0,i));
    double cf = std::sqrt(0.5*(pow(cs,2)+pow(ca,2)+std::sqrt(pow(pow(cs,2)+pow(ca,2),2)-4*(pow(cs,2)*pow(Bx,2)/v(0,i)))));
    a(i) =  std::sqrt(total_velocity) + cf;
//std::cout << "this is cs: " << cs << " this is ca:" << ca << " this is cf " << cf <<  " pressure :" << pressure << " energy:" <<v(2,i)<< "\n";
  }


  double max_a = *std::max_element(a.begin(),a.end());
  double time_step = (c * dx) / fabs(max_a);
  return time_step ;
}

MatrixXd Solver::euler_flux(MatrixXd v){

// Compute the numerical flux of each solution  

  for (int i = 0 ; i< cells+2;i++){
  
    double velocity_x = v(1,i)/v(0,i);
    double velocity_y = v(2,i)/v(0,i);
    double velocity_z = v(3,i)/v(0,i);
    double total_velocity = pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2);
    double Bsqrd = pow(v(6,i),2)+pow(v(5,i),2)+pow(Bx,2); 
    double pressure = ((v(4,i)-(0.5*v(0,i)*total_velocity) - 0.5*(Bsqrd))*(gamma-1)) ;

    flux_initial(0,i) = v(1,i);// density flux                                     
    flux_initial(1,i) = v(1,i)*velocity_x + pressure + 0.5*Bsqrd - pow(Bx,2);
    flux_initial(2,i) = v(2,i)*velocity_x - Bx*v(5,i);
    flux_initial(3,i) = v(3,i)*velocity_x - Bx*v(6,i);
    flux_initial(4,i) = (v(4,i) + pressure + 0.5*Bsqrd)*velocity_x - (velocity_x*Bx + velocity_y*v(5,i)+velocity_z*v(6,i))*Bx;
    flux_initial(5,i) = v(5,i)*velocity_x - Bx*velocity_y;
    flux_initial(6,i) = v(6,i)*velocity_x - Bx*velocity_z;
  }
  return flux_initial;
}

double Solver::minbee(double r){

  double epsalon;

        if(r <= 0){
          epsalon = 0.;
        }else if (r > 0 && r <=1){

          epsalon = r;
        }else if (r > 1)
          {epsalon = std::min(1.,(2./(1.+r)));
        }

return epsalon;
}

double Solver::Van_Albada(double r ){

  double epsalon;

  if(r <= 0){
          epsalon = 0.;
        }else 
          {epsalon = std::min((r*(1+r))/(1+pow(r,2)),(2./(1.+r)));
        }

return epsalon;
}

void Solver::simulate(){
 
  initialise_input();

  double t = 0;

  do{

   for(int i=0; i<7; i++){
      v(i,0) = v(i,1);
      v(i,cells+1) = v(i,cells);
    }

   dt = compute_time_step();
   t = t+dt;

   //euledelta_upwind(k,i)/delta_downwind(k,i)r_flux(v); // calculate the numerical flux for the solution vector.
  

    for(int k=0; k < v.rows(); k++){ // set up the half time step matrix needed for the FORCE flux. 

    for (int i=1;i<cells+2;i++){
      delta_upwind(k,i) = v(k,i) - v(k,i-1);
    }
    delta_upwind(k,0) = delta_upwind(k,1);


    for(int i=0; i < cells+1; i++){
        delta_downwind(k,i) = v(k,i+1)-v(k,i);
    }

    delta_downwind(k,cells+1) = delta_downwind(k,cells);

    for(int i=0; i<cells+2;i++){   
        delta(k,i) = 0.5*delta_upwind(k,i) + 0.5*delta_downwind(k,i);

        if(std::isnan(delta_upwind(k,i)/delta_downwind(k,i)) == 1){
          r(k,i) = 0;
          }else{
            r(k,i) = delta_upwind(k,i)/delta_downwind(k,i);
          }

        epsalon(k,i) = minbee(r(k,i));
        v_L(k,i) = v(k,i) - 0.5*epsalon(k,i)*delta(k,i); 
        v_R(k,i) = v(k,i) + 0.5*epsalon(k,i)*delta(k,i); 
      }
     }

    v_L_flux = euler_flux(v_L);
    v_R_flux = euler_flux(v_R);

    for(int k=0; k < v.rows(); k++){ // set up the half time step matrix needed for the FORCE flux. 
      for(int i=0; i < cells+2; i++){

        v_L_half(k,i) = v_L(k,i) - 0.5*(dt/dx)*(v_R_flux(k,i)-v_L_flux(k,i));

        v_R_half(k,i) = v_R(k,i) - 0.5*(dt/dx)*(v_R_flux(k,i)-v_L_flux(k,i));
      }
    }

    v_L_flux = euler_flux(v_L_half);
    v_R_flux = euler_flux(v_R_half);

    for(int i =0 ; i < cells+1 ; i ++){

    double velocity_x_l = v_R_half(1,i)/v_R_half(0,i);
    double velocity_y_l = v_R_half(2,i)/v_R_half(0,i);
    double velocity_z_l = v_R_half(3,i)/v_R_half(0,i);

    double total_velocity = pow(velocity_x_l,2)+pow(velocity_y_l,2)+pow(velocity_z_l,2);
    double Bsqrd_l = pow(v_R_half(6,i),2)+pow(v_R_half(5,i),2)+pow(Bx,2); 
    double pressure_l = ((v_R_half(4,i)-(0.5*v_R_half(0,i)*total_velocity) - 0.5*Bsqrd_l)*(gamma-1));
    double cs = std::sqrt((gamma * (pressure_l)) /v_R_half(0,i));
    double ca  =std::sqrt(Bsqrd_l)/std::sqrt(v_R_half(0,i));
    double cf_l = std::sqrt(0.5*(pow(cs,2)+pow(ca,2)+std::sqrt(pow(pow(cs,2)+pow(ca,2),2)-4*(pow(cs,2)*pow(Bx,2)/v_R_half(0,i)))));
    double total_pressure_l = pressure_l + 0.5*Bsqrd_l;
// Right state cf

    double velocity_x_r = v_L_half(1,i+1)/v_L_half(0,i+1);
    double velocity_y_r = v_L_half(2,i+1)/v_L_half(0,i+1);
    double velocity_z_r = v_L_half(3,i+1)/v_L_half(0,i+1);
    double total_velocity_r = pow(velocity_x_r,2)+pow(velocity_y_r,2)+pow(velocity_z_r,2);
    double Bsqrd_r = pow(v_L_half(6,i+1),2)+pow(v_L_half(5,i+1),2)+pow(Bx,2); 
    double pressure_r = ((v_L_half(4,i+1)-(0.5*v_L_half(0,i+1)*total_velocity_r) - 0.5*Bsqrd_r)*(gamma-1));
    cs = std::sqrt((gamma * (pressure_r)) /v_L_half(0,i+1));
    ca = std::sqrt(Bsqrd_r)/std::sqrt(v_L_half(0,i+1));
    double cf_r = std::sqrt(0.5*(pow(cs,2)+pow(ca,2)+std::sqrt(pow(pow(cs,2)+pow(ca,2),2)-4*(pow(cs,2)*pow(Bx,2)/v_L_half(0,i+1)))));
    double total_pressure_r = pressure_r + 0.5*Bsqrd_r;
// -----------------------------------------------------------------------------------------------------------------



    double s_l = std::min(velocity_x_r,velocity_x_l) - std::max(cf_r,cf_l);
    double s_r = std::max(velocity_x_r,velocity_x_l) + std::max(cf_r,cf_l);

    double Bx_star_r = Bx;
    double By_star_r = (s_r*v_L_half(5,i+1) - s_l*v_R_half(5,i) - v_L_flux(5,i+1) + v_R_flux(5,i))/(s_r-s_l);
    double Bz_star_r = (s_r*v_L_half(6,i+1) - s_l*v_R_half(6,i) - v_L_flux(6,i+1) + v_R_flux(6,i))/(s_r-s_l);


    double rho_v_x_hll = (s_r*v_L_half(1,i+1) - s_l*v_R_half(1,i) + v_R_flux(1,i)-v_L_flux(1,i+1))/(s_r-s_l);
    double rho_v_y_hll = (s_r*v_L_half(2,i+1) - s_l*v_R_half(2,i) + v_R_flux(2,i)-v_L_flux(2,i+1))/(s_r-s_l);
    double rho_v_z_hll = (s_r*v_L_half(3,i+1) - s_l*v_R_half(3,i) + v_R_flux(3,i)-v_L_flux(3,i+1))/(s_r-s_l);

    double rho_hll = (s_r*v_L_half(0,i+1)- s_l*v_R_half(0,i) + v_R_flux(0,i) - v_L_flux(0,i+1))/(s_r-s_l); 

    double v_x_hll = rho_v_x_hll/rho_hll;
    double v_y_hll = rho_v_y_hll/rho_hll;
    double v_z_hll = rho_v_z_hll/rho_hll;

    double v_star = ((v_L_half(1,i+1)*(s_r - velocity_x_r)) - v_R_half(1,i)*(s_l-velocity_x_l) + total_pressure_l - total_pressure_r - pow(Bx,2) + pow(Bx,2))/ (v_L_half(0,i+1)*(s_r - velocity_x_r) - v_R_half(0,i)*(s_l-velocity_x_l));
    double p_star = (v_R_half(0,i)*(s_l-velocity_x_l)*(v_star-velocity_x_l)) + total_pressure_l - pow(Bx,2) + pow(Bx_star_r,2);
    double B_v_dot_star = Bx_star_r*v_x_hll + By_star_r*v_y_hll + Bz_star_r*v_z_hll;
    double B_v_dot_r = Bx*velocity_x_r + v_L_half(5,i+1)*velocity_y_r + v_L_half(6,i+1)*velocity_z_r; 
    double B_v_dot_l = Bx*velocity_x_l + v_R_half(5,i)*velocity_y_l + v_R_half(6,i)*velocity_z_l; 

    star_r(0,i) = v_L_half(0,i+1)*((s_r - velocity_x_r)/(s_r - v_star));
    star_r(1,i) = star_r(0,i)*v_star;
    star_r(2,i) = v_L_half(2,i+1)*((s_r - velocity_x_r)/(s_r - v_star)) - (Bx_star_r*By_star_r - Bx*v_L_half(5,i+1))/(s_r - v_star);
    star_r(3,i) = v_L_half(3,i+1)*((s_r - velocity_x_r)/(s_r - v_star)) - (Bx_star_r*Bz_star_r - Bx*v_L_half(6,i+1))/(s_r - v_star);
    star_r(4,i) = v_L_half(4,i+1)*((s_r - velocity_x_r)/(s_r - v_star)) + (p_star*v_star -  total_pressure_r*velocity_x_r - (Bx_star_r*(B_v_dot_star) - Bx*(B_v_dot_r)))/(s_r-v_star);
    star_r(5,i) = By_star_r;
    star_r(6,i) = Bz_star_r;


    star_l(0,i) = v_R_half(0,i)*((s_l - velocity_x_l)/s_l - v_star);
    star_l(1,i) = star_l(0,i)*v_star;
    star_l(2,i) = v_R_half(2,i)*((s_l - velocity_x_l)/(s_l - v_star)) - (Bx_star_r*By_star_r - Bx*v_R_half(5,i))/(s_l - v_star);
    star_l(3,i) = v_R_half(3,i)*((s_l - velocity_x_l)/(s_l - v_star)) - (Bx_star_r*Bz_star_r - Bx*v_R_half(6,i))/(s_l - v_star);
    star_l(4,i) = v_R_half(4,i)*((s_l - velocity_x_l)/(s_l - v_star)) + (p_star*v_star -  total_pressure_l*velocity_x_l - (Bx_star_r*(B_v_dot_star) - Bx*(B_v_dot_l)))/(s_l-v_star);
    star_l(5,i) = By_star_r;
    star_l(6,i) = Bz_star_r;


    for(int k=0; k<7;k++)
    if(0<=s_l){

      flux(k,i) = v_R_flux(k,i);
    }else if (s_l<=0 && 0<=v_star){
      flux(k,i) = v_R_flux(k,i) + s_l*(star_l(k,i) - v_R_half(k,i));
    }else if(v_star <= 0 && 0<=s_r){
      flux(k,i) = v_L_flux(k,i+1)+s_r*(star_r(k,i)-v_L_half(k,i+1));
    }else if(s_r<=0){
      flux(k,i) = v_L_flux(k,i+1);
    }
    }

  for (int k=0; k < v.rows(); k++){
   
      for (int i=1; i < cells+1; i++ ){


       v_new(k,i) = v(k,i) - (dt/dx) * (flux(k,i)-flux(k,i-1));
      
      }
    }


 for(int i =0; i<7; i++){
      v_new(i,0) = v_new(i,1);
      v_new(i,cells+1) = v_new(i,cells);
    }


 v = v_new;
 //break;
  } while (t < final_time);
}

void Solver::save_plot(){

  std::ofstream output("output.dat");

  for(int i = 1; i < cells+1; i++ ){

    double x = x0 + (i-1)*dx;
    double Bsqrd = pow(v(6,i),2)+pow(v(5,i),2)+pow(Bx,2); 
    double velocity_x = v(1,i)/v(0,i);
    double velocity_y = v(2,i)/v(0,i);
    double velocity_z = v(3,i)/v(0,i);
    output << x << " " << v(0,i) << " " << v(1,i)/v(0,i) << " " << v(2,i)/v(0,i) << " " << v(3,i)/v(0,i) << " " <<  ((v(4,i)-(0.5*v(0,i)*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) - 0.5*Bsqrd)*(gamma-1)) << " " << Bx << " " << v(5,i) << " " << v(6,i) << "\n";
  
  }
}

int main(){
 Solver solver;
 solver.simulate(); 
 solver.save_plot();
}
