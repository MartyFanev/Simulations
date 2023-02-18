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
};

Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.9;
  cells = 200;
  a = 1.;
  final_time = 80.;
  x0 = 0.;
  x1 = 800.;
  dx = (x1-x0)/cells;
  gamma = 1.4; // use gamma for stiffened water
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
  llf.resize(7,v.cols()+1);
  v_L.resize(7,v.cols());
  v_R.resize(7,v.cols());
  rychtmer.resize(7,v.cols()+1);
  delta_upwind.resize(7,v.cols());
  delta_downwind.resize(7,v.cols());
  delta.resize(7,v.cols());
  flux_initial.resize(7,cells+2);
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
      std::cout << pressure/(gamma-1);
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
      double pressure=0.1;
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

    double velocity_x = v(1,i)/v(0,i);
    double velocity_y = v(2,i)/v(0,i);
    double velocity_z = v(3,i)/v(0,i);
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
   std::cout <<dt;
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


 for(int i =0; i<7; i++){
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
