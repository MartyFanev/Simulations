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
    MatrixXd v_R_flux;
    MatrixXd v_L_flux;
    MatrixXd llf;
    MatrixXd flux;
    double gamma;
};

Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.9;
  cells = 200;
  a = 1.;
  final_time = .012;
  x0 = 0.;
  x1 = 1.;
  dx = (x1-x0)/cells;
  gamma = 1.4;
  v.resize(3,cells+2); // set up the vector for the solution
  v_L_flux.resize(3,v.cols());
  v_R_flux.resize(3,v.cols());
  v_new.resize(3,cells+2); // set up the placement vector for the solutions
  epsalon.resize(3,v.cols());
  half_time_step.resize(3,v.cols());
  v_L_half.resize(3,v.cols());
  v_R_half.resize(3,v.cols());
  r.resize(3,v.cols());
  llf.resize(3,v.cols()+1);
  v_L.resize(3,v.cols());
  v_R.resize(3,v.cols());
  rychtmer.resize(3,v.cols()+1);
  delta_upwind.resize(3,v.cols());
  delta_downwind.resize(3,v.cols());
  delta.resize(3,v.cols());
  flux_initial.resize(3,cells+2);
}

void Solver::initialise_input(){
  std::ofstream input("input.dat");

// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < v.cols(); i++){
    double x = x0 + (i-0.5)*dx;

    if(x < 0.5){

      double pressure =0.01;
      double velocity = 0.0;
      double density = 1.0;
      double energy = (pressure/(gamma-1)) + (0.5*density*pow(velocity,2));

      v(0,i) = density;
      v(1,i) = velocity * density; // the momentum value 
      v(2,i) = energy;

    }else if (x>0.5){

      double pressure = 100.0;
      double velocity =0.0;
      double density = 1.0;
      double energy = pressure/(gamma-1) + 0.5*density*pow(velocity,2);

      v(0,i) = density;
      v(1,i) = velocity * density;// momentum 
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

    double velocity = abs_v(1,i)/abs_v(0,i);
    double pressure = (v(2,i)-(0.5*v(0,i)*pow(velocity,2)))*(gamma-1); 

    double cs = std::sqrt((1.4 * pressure /v(0,i)));
    a(i) =  velocity + cs;

  }


  double max_a = *std::max_element(a.begin(),a.end());
  double time_step = (c * dx) / fabs(max_a);
  return time_step ;
}

MatrixXd Solver::euler_flux(MatrixXd v){

// Compute the numerical flux of each solution  

  for (int i = 0 ; i< cells+2;i++){
  
    double velocity = v(1,i)/v(0,i);
    double pressure = (v(2,i)-(0.5*v(0,i)*pow(velocity,2)))*(gamma-1); 

    flux_initial(0,i) = v(1,i);// density flux
                                         
    flux_initial(1,i) = (v(0,i)*pow(velocity,2)) + pressure; //velocity flux 

    flux_initial(2,i) = (v(2,i)+pressure)*velocity;
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
std::cout << r << "\n";



    v_L_flux = euler_flux(v_L_half);
    v_R_flux = euler_flux(v_R_half);


    for (int k=0; k < flux.rows();k++){
     for (int i=0; i < cells+1 ;i++){

     half_time_step(k,i) = 0.5*(v_R_half(k,i) + v_L_half(k,i+1)) - 0.5*(dt/dx)*(-v_R_flux(k,i)+v_L_flux(k,i+1));

     llf(k,i)  = (0.5 * (dx/dt) * (v_R_half(k,i)-v_L_half(k,i+1))) + (0.5*(v_R_flux(k,i) + v_L_flux(k,i+1)));
       

     }
    }


    flux_initial = euler_flux(half_time_step);

    for (int k=0; k < flux.rows();k++){
    for (int i=0; i < cells+1 ;i++){
      rychtmer(k,i) = flux_initial(k,i);
      }
    }



  flux = 0.5*(llf+rychtmer);


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
  } while (t < final_time);
}

void Solver::save_plot(){

  std::ofstream output("output.dat");

  for(int i = 1; i < cells+1; i++ ){

    double x = x0 + (i-1)*dx;
    output << x << " " << v(0,i) << " " << v(1,i)/v(0,i) << " " <<  (v(2,i)-(0.5*v(0,i)*pow((v(1,i)/v(0,i)),2)))*(gamma-1) << "\n";
  
  }
}

int main(){
 Solver solver;
 solver.simulate(); 
 solver.save_plot();
}
