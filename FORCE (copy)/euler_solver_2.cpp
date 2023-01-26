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
    void euler_flux();
    void save_plot();
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
    double gamma;
};


Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.8;
  cells = 2000;
  a = 1.;
  final_time = 0.25;
  x0 = 0.;
  x1 = 1.;
  dx = (x1-x0)/cells;
  gamma = 1.4;
}

void Solver::initialise_input(){
  std::ofstream input("input.dat");

  v.resize(3,cells+2); // set up the vector for the solution
  v_new.resize(3,cells+2); // set up the placement vector for the solutions


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < v.cols(); i++){
    double x = x0 + (i-0.5)*dx;

     if(x < 0.4){

      double pressure =1.0;
      double velocity = 0.0;
      double density = 1.0;
      double energy = (pressure/(gamma-1)) + (0.5*density*pow(velocity,2));

      v(0,i) = density;
      v(1,i) = velocity * density; // the momentum value 
      v(2,i) = energy;

    }else if (x>0.4){

      double pressure = 0.1;
      double velocity =0.0;
      double density = 0.125;
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




void Solver::euler_flux(){

// Compute the numerical flux of each solution  

  flux_initial.resize(3,cells+2);

  for (int i =0 ; i< v.cols();i++){
  
    double velocity = v(1,i)/v(0,i);
    double pressure = (v(2,i)-(0.5*v(0,i)*pow(velocity,2)))*(gamma-1); 

    flux_initial(0,i) = v(1,i);// density flux
                                         
    flux_initial(1,i) = (v(0,i)*pow(velocity,2)) + pressure; //velocity flux 

    flux_initial(2,i) = (v(2,i)+pressure)*velocity;
  }
}


void Solver::simulate(){
 
  initialise_input();
  double t = 0;

  MatrixXd flux;
  flux.resize(3,v.cols()+1);

  MatrixXd llf;
  llf.resize(3,v.cols()+1);

  MatrixXd half_time_step;
  half_time_step.resize(3,v.cols());

  MatrixXd rychtmer;
  rychtmer.resize(3,v.cols()+1);

  do{

   // Set up boundary conditions

   for(int i =0; i<3; i++){
      v(i,0) = v(i,1);
      v(i,cells+1) = v(i,cells);
    }

   std::cout << v;

   //compute the time step

   dt = compute_time_step();
   t = t+dt;

  euler_flux(); // calculate the numerical flux for the solution vector.

  for(int k =0; k < v.rows(); k++){ // set up the half time step matrix needed for the FORCE flux. 
    for(int i=0; i < cells+1; i++){
      half_time_step(k,i) = 0.5*(v(k,i) + v(k,i+1)) - 0.5*(dt/dx)*(flux_initial(k,i+1)-flux_initial(k,i));
    }
  }


//build the llf and rychtmer fluxes and the overall flux matrix

  for (int k=0; k < flux.rows();k++){

    for (int i=0; i < cells+1 ;i++){

      double half_velocity = half_time_step(1,i)/half_time_step(0,i);
      double half_pressure = (half_time_step(2,i)-(0.5*half_time_step(0,i)*pow(half_velocity,2)))*(gamma-1);


       llf(k,i)  = (0.5 * (dx/dt) * (v(k,i)-v(k,i+1))) + (0.5*(flux_initial(k,i) + flux_initial(k,i+1)));

       rychtmer(0,i) = half_time_step(1,i);


       rychtmer(1,i) = (half_time_step(0,i) * pow(half_velocity,2)) + half_pressure;


       rychtmer(2,i) = (half_time_step(2,i) + half_pressure) * half_velocity;
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
 std::cout << v_new;

  }while (t < final_time);
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
