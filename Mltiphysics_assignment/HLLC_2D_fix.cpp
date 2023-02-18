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
  c = 0.8;
  cells = 100;
  a = 1.;
  final_time = 0.25;
  x0 = 0.;
  x1 = 1.;
  dx = (x1-x0)/cells;
  gamma = 1.4; // use gamma for stiffened water
  p_inf = 6e8;
  v.resize(4,cells+2); // set up the vector for the solution
  v_L_flux.resize(4,v.cols());
  v_R_flux.resize(4,v.cols());
  v_new.resize(4,cells+2); // set up the placement vector for the solutions
  epsalon.resize(4,v.cols());
  half_time_step.resize(4,v.cols());
  v_L_half.resize(4,v.cols());
  v_R_half.resize(4,v.cols());
  r.resize(4,v.cols());
  v_L.resize(4,v.cols());
  v_R.resize(4,v.cols());
  delta_upwind.resize(4,v.cols());
  delta_downwind.resize(4,v.cols());
  delta.resize(4,v.cols());
  flux_initial.resize(4,cells+2);
  flux.resize(4,cells+2);
  hllc.resize(4,v.cols());
  star_l.resize(4,v.cols());
  star_r.resize(4,v.cols());
}

void Solver::initialise_input(){
  std::ofstream input("input.dat");

// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < v.cols(); i++){
    double x = x0 + (i-0.5)*dx;

    if(x < 0.5){

      
      double density = 1.0;
      double velocity_x = -0.0;
      double pressure=1.0;
      double energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)));

      v(0,i) = density;
      v(1,i) = velocity_x * density; // the momentum value y*density
      v(2,i) = energy;

    }else if (x>0.5){

      double density = 0.125;
      double velocity_x = 0.0;
      double pressure=0.1; // this variable will be a standalone for total_pressure

      double energy = (pressure/(gamma-1)) + (0.5*density*pow(velocity_x,2));

      v(0,i) = density;
      v(1,i) = velocity_x * density; // the momentum value y*density
      v(2,i) = energy;

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
    //double velocity_y = abs_v(2,i)/abs_v(0,i);
    double total_velocity = pow(velocity_x,2);
    double pressure = ((v(2,i)-(0.5*v(0,i)*total_velocity))*(gamma-1));

    double cs = std::sqrt((gamma * (pressure)) /v(0,i));
    a(i) =  std::sqrt(total_velocity) + cs;
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
    //double velocity_y = v(2,i)/v(0,i);

    double total_velocity = pow(velocity_x,2);

    double pressure = ((v(2,i)-(0.5*v(0,i)*total_velocity))*(gamma-1));

      flux_initial(0,i) = v(1,i);// density flux
                                         
      flux_initial(1,i) = (v(0,i)*pow(velocity_x,2)) + pressure; //velocity flux 

      flux_initial(2,i) = (v(2,i)+pressure)*velocity_x;

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

   for(int i=0; i<3; i++){
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

    double total_velocity_l = pow(velocity_x_l,2);
    double pressure_l = ((v_R_half(2,i)-(0.5*v_R_half(0,i)*total_velocity_l))*(gamma-1));
    double cs_l = std::sqrt((gamma * (pressure_l)) /v_R_half(0,i));

    double velocity_x_r = v_L_half(1,i+1)/v_L_half(0,i+1);
    double total_velocity_r = pow(velocity_x_r,2);
    double pressure_r = ((v_L_half(2,i+1)-(0.5*v_L_half(0,i+1)*total_velocity_r))*(gamma-1));
    double cs_r = std::sqrt((gamma * (pressure_r)) /v_L_half(0,i+1));
    
// -----------------------------------------------------------------------------------------------------------------
  


    double s_r = std::min(abs(velocity_x_r) + cs_r,abs(velocity_x_l)+cs_l);
    double s_l = -std::max(abs(velocity_x_r) + cs_r,abs(velocity_x_l)+cs_l);



    double v_star = (pressure_r - pressure_l + v_R_half(1,i)*(s_l-velocity_x_l) - v_L_half(1,i+1)*(s_r-velocity_x_r))/(v_R_half(0,i)*(s_l-velocity_x_l) - v_L_half(0,i+1)*(s_r-velocity_x_r));
    
    star_r(0,i) = v_L_half(0,i+1)*((s_r - velocity_x_r)/(s_r - v_star));
    star_r(1,i) = star_r(0,i)*v_star;
    star_r(2,i) = star_r(0,i) * ((v_L_half(2,i+1))/(v_L_half(0,i+1)) + (v_star-velocity_x_r)*(v_star + ((pressure_r)/(v_L_half(0,i+1)*(s_r-velocity_x_r)))));


    star_l(0,i) = v_R_half(0,i)*((s_l - velocity_x_l)/(s_l - v_star));
    star_l(1,i) = star_l(0,i)*v_star;
    star_l(2,i) = star_l(0,i)*(((v_R_half(2,i))/(v_R_half(0,i))) + (v_star-velocity_x_l)*(v_star + ((pressure_l)/(v_R_half(0,i)*(s_l-velocity_x_l)))));


    for(int k=0; k<3;k++)
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


 for(int i =0; i<3; i++){
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
    //double Bsqrd = pow(v(6,i),2)+pow(v(5,i),2)+pow(Bx,2); 
    double velocity_x = v(1,i)/v(0,i);
    //double velocity_y = v(2,i)/v(0,i);
    //double velocity_z = v(3,i)/v(0,i);
    output << x << " " << v(0,i) <<  " " << velocity_x << " " << (v(2,i)-0.5*v(0,i)*(pow(velocity_x,2)))*(gamma-1)  << " " << (v(2,i)-0.5*v(0,i)*pow(velocity_x,2))/v(0,i) << " " <<"\n"; //  v(1,i)/v(0,i) << " " << v(2,i)/v(0,i) << " " << v(3,i)/v(0,i) << " " <<  ((v(4,i)-(0.5*v(0,i)*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) - 0.5*Bsqrd)*(gamma-1)) << " " << Bx << " " << v(5,i) << " " << v(6,i) << "\n";
  
  }
}

int main(){
 Solver solver;
 solver.simulate(); 
 solver.save_plot();
}
