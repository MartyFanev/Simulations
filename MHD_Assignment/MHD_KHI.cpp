#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Solver {
  public:
    void print_stuff();
    void initialise_input();
    Solver();
    double compute_time_step();
    void simulate();
    void initialise_input_explosion();
    void initialise_MHD();
    void initialise_KHI();
    double Van_Albada(double r);
    Tensor<double,3> euler_flux(Tensor<double,3> v);
    Tensor<double,3> euler_flux_y(Tensor<double,3> v);
    Tensor<double,3> euler_flux_x(Tensor<double,3> v);
    Tensor<double,3> flux_all(Tensor<double,3> v,char option);
    Tensor<double,3> W_to_U(Tensor<double,3> w);
    Tensor<double,3> U_to_W(Tensor<double,3> v);
    void initialise_Sod();
    void save_plot();
    Tensor<double,3> periodic_boundary(Tensor<double,3> v);
    Tensor<double,3> KHI_boundary(Tensor<double,3 >w);
    void SLIC();

  private:
    double c;
    double a;
    double final_time;
    double x0;
    double x1;
    double dx;
    double dy;
    double y0;
    double y1;
    double dt;
    double pi;
    int cells;
    Tensor<double,3>  v;
    Tensor<double,3>  v_new;
     Tensor<double,3>  flux_initial;
     Tensor<double,3>  half_time_step;
     Tensor<double,3>  epsalon;
     Tensor<double,3>  r;
     Tensor<double,3>  v_R_half;
     Tensor<double,3>  v_L_half;
     Tensor<double,3>  delta;
     Tensor<double,3>  delta_downwind;
     Tensor<double,3>  delta_upwind;
     Tensor<double,3>  rychtmer;
     Tensor<double,3>  v_R;
     Tensor<double,3>  v_R_flux;
     Tensor<double,3>  v_L;
     Tensor<double,3>  v_L_flux;
     Tensor<double,3>  llf;
     Tensor<double,3>  flux;
      Tensor<double,3>  w;
    double gamma;
};


Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.8;
  cells = 200;
  a = 1.;
  final_time = 0.5;
  x0 = 0.;
  y0 = -0.;
  x1 = 1.;
  y1 = 1.;
  dx = (x1-x0)/cells;
  dy = (y1-y0)/cells;
  std::cout << dx;
  gamma = 5./3.;
  pi=2*std::acos(0.0);
  half_time_step = Tensor<double,3>((cells+4),(cells+4),8);
  flux = Tensor<double,3>((cells+5),(cells+5),8);
  llf = Tensor<double,3>((cells+5),(cells+5),8);
  v_L = Tensor<double,3>((cells+4),(cells+4),8);
  v_R = Tensor<double,3>((cells+4),(cells+4),8);
  v_R_flux = Tensor<double,3>((cells+4),(cells+4),8);
  v_L_flux = Tensor<double,3>((cells+4),(cells+4),8);
  rychtmer = Tensor<double,3>((cells+5),(cells+5),8);
  delta_upwind = Tensor<double,3>((cells+4),(cells+4),8);
  delta_downwind = Tensor<double,3>((cells+4),(cells+4),8);
  delta = Tensor<double,3>((cells+4),(cells+4),8);
  v_L_half = Tensor<double,3>((cells+4),(cells+4),8);
  v_R_half = Tensor<double,3>((cells+4),(cells+4),8);
  r = Tensor<double,3>((cells+4),(cells+4),8);
  epsalon = Tensor<double,3>((cells+4),(cells+4),8);
  half_time_step = Tensor<double,3>((cells+4),(cells+4),8);
  flux_initial = Tensor<double,3>(cells+4,cells+4,8);
  v = Tensor<double,3>(cells+4,cells+4,8); // set up the vector for the solution
  v_new = Tensor<double,3>(cells+4,cells+4,8);
  w = Tensor<double,3>(cells+4,cells+4,8);
}

void Solver::initialise_input(){
  std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double velocity_z;
   double density;
   double energy;
   double By;
   double Bz;
   double Bx;
   double Bsqrd;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;

      if(x < 0.5 && y < 0.5){

        pressure =1.0;
        velocity_x = -0.75;
        velocity_y= 0.5;
        velocity_z =0.;
        Bx = 0.;
        By=0.;
        Bz=0.;
        density = 1.0;
        Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      }else if (x>0.5 && y <0.5){

        pressure = 1.;
        velocity_x =-0.75;
        velocity_y = -0.5;
        velocity_z =0.;
        Bx = 0.;
        By=0.;
        Bz=0.;
        density = 3.;
        Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      }else if(x < 0.5 && y > 0.5){

        pressure = 1.;
        velocity_x =0.75;
        velocity_y = 0.5;
        velocity_z =0.;
        Bx = 0.;
        By=0.;
        Bz=0.;
        density = 2.;
        Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      }else if(x > 0.5 && y > 0.5){

        pressure = 1.;
        velocity_x =0.75;
        velocity_y = -0.5;
        velocity_z =0.;
        Bx = 0.;
        By=0.;
        Bz=0.;
        density = 1.;
        Bsqrd = pow(By,2)+pow(Bx,2)+pow(Bz,2);
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2))) + 0.5*(Bsqrd);

      }
        v(i,j,0) = density;
        v(i,j,1) = velocity_x * density; // the momentum value 
        v(i,j,2) = energy;
        v(i,j,3) = velocity_y*density;
        v(i,j,4) = velocity_z*density;
        v(i,j,5) = Bx;
        v(i,j,6) = By;
        v(i,j,7) = Bz;
    
      input << x << " " << y << " " << v(i,j,2) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }

}

void Solver::initialise_MHD(){
  std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double velocity_z;
   double By;
   double Bz;
   double Bx;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;


        w(i,j,0) = gamma*gamma;
        w(i,j,1) = -std::sin(2*pi*y);
        w(i,j,2) = std::sin(2*pi*x);
        w(i,j,3) =0.;
        w(i,j,4) = gamma;
        w(i,j,5) = -std::sin(2*pi*y);
        w(i,j,6)=std::sin(4*pi*x);
        w(i,j,7)=0.;
    
      input << x << " " << y << " " << w(i,j,0) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }

}

Tensor<double,3> Solver::W_to_U(Tensor<double,3> w){


for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

      double Bsqrd = pow(w(i,j,5),2)+pow(w(i,j,6),2)+pow(w(i,j,7),2);
      double energy = (w(i,j,4)/(gamma-1)) + (0.5*w(i,j,0)*(pow(w(i,j,1),2)+pow(w(i,j,2),2)+pow(w(i,j,3),2))) + 0.5*(Bsqrd);


        v(i,j,0) = w(i,j,0);
        v(i,j,1) = w(i,j,1) * w(i,j,0); // the momentum value 
        v(i,j,2) = energy;
        v(i,j,3) = w(i,j,2)*w(i,j,0);
        v(i,j,4) = w(i,j,3)*w(i,j,0);
        v(i,j,5) = w(i,j,5);
        v(i,j,6) = w(i,j,6);
        v(i,j,7) = w(i,j,7);

    }
  } 

  return v;

}

Tensor<double,3> Solver::U_to_W(Tensor<double,3> v){

for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){


     double velocity_x= (v(i,j,1)/v(i,j,0));
     double velocity_y = v(i,j,3)/v(i,j,0);
     double velocity_z = v(i,j,4)/v(i,j,0);



     double Bsqrd = pow(v(i,j,5),2) + pow(v(i,j,6),2) + pow(v(i,j,7),2);


     double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2));
     double pressure = (v(i,j,2)-kinetic_energy-0.5*Bsqrd)*(gamma-1); 



        w(i,j,0) = v(i,j,0);
        w(i,j,1) = velocity_x; // the momentum value 
        w(i,j,2) = velocity_y;
        w(i,j,3) = velocity_z;
        w(i,j,4) = pressure;
        w(i,j,5) = v(i,j,5);
        w(i,j,6) = v(i,j,6);
        w(i,j,7) = v(i,j,7);

    }
  } 

  return w;

}

void Solver::initialise_KHI(){

    std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double velocity_z;
   double By;
   double Bz;
   double Bx;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;


        w(i,j,0) = 1; // density
        w(i,j,1) = 0.5*std::tanh(20.*y); // velocity_x
        w(i,j,2) = 0.1*std::sin(2*pi*x)*std::exp(-pow(y,2)/pow(0.1,2));// velocity_y
        w(i,j,3) = 0.; // velocity z
        w(i,j,4) = 1./gamma; // pressure
        w(i,j,5) = 0.1*std::sqrt(w(i,j,0))*std::cos(pi/3.); // Bx
        w(i,j,6) = 0.;//By
        w(i,j,7) = 0.1*std::sqrt(w(i,j,0))*std::sin(pi/3.);//Bz

      input << x << " " << y << " " << w(i,j,0) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }

}

void Solver::initialise_Sod(){
  std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;

      if(y < 0.5){

        pressure =1.0;
        velocity_x = 0.0;
        velocity_y= 0.0;
        density = 1.0;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));


      }else if(y > 0.5){

        pressure = 0.1;
        velocity_x =0.;
        velocity_y = 0.;
        density = 0.125;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

      }

      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
      input << x << " " << y << " " << v(i,j,2) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }

}

void Solver::initialise_input_explosion(){


  std::ofstream input("input.dat");

    double pressure;
    double velocity_x;
    double velocity_y;
    double density;
    double energy;

    for(int i=0; i < cells+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+4; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;
        double r = std::sqrt(pow(x-1,2)+pow(y-1,2));

        if(r<=0.4){

        pressure = 1.;
        velocity_x =0.;
        velocity_y = 0.;
        density = 1.;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

        }else{


        pressure = 0.1;
        velocity_x =0.;
        velocity_y = 0.;
        density = 0.125;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

        }


        v(i,j,0) = density;
        v(i,j,1) = velocity_x * density; // the momentum value 
        v(i,j,2) = energy;
        v(i,j,3) = velocity_y*density;

        input << x << " " << y << " " << v(i,j,2) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";

      }
    }

}

Tensor<double,3> Solver::periodic_boundary(Tensor<double,3 >v){

     for(int k=0; k<8; k++){
    for(int j=0; j<cells+2; j++){
      v(0,j,k) = v(cells,j,k);
      v(1,j,k) = v(cells+1,j,k);

      v(cells+2,j,k) = v(2,j,k);
      v(cells+3,j,k) = v(3,j,k);

    }


    for(int i=0; i<cells+2; i++){
      v(i,0,k) = v(i,cells,k);
      v(i,1,k) = v(i,cells+1,k);

      v(i,cells+2,k) = v(i,2,k);
      v(i,cells+3,k) = v(i,3,k);
    }
   }
return v;
}

Tensor<double,3> Solver::KHI_boundary(Tensor<double,3 >w){


    for(int k=0; k<8; k++){
    for(int j=0; j<cells+4; j++){
      w(0,j,k) = w(cells,j,k);
      w(1,j,k) = w(cells+1,j,k);

      w(cells+2,j,k) = w(2,j,k);
      w(cells+3,j,k) = w(3,j,k);

    }


    for(int i=0; i<cells+4; i++){
      
      w(i,1,k) = w(i,2,k);
      w(i,0,k) = w(i,3,k);

      w(i,cells+2,k) = w(i,cells+1,k);
      w(i,cells+3,k) = w(i,cells,k);


      w(i,1,2) = -w(i,2,2);
      w(i,0,2) = -w(i,3,2);

      w(i,cells+2,2) = -w(i,cells+1,2);
      w(i,cells+3,2) = -w(i,cells,2);

      w(i,1,6) = -w(i,2,6);
      w(i,0,6) = -w(i,3,6);

      w(i,cells+2,6) = -w(i,cells+1,6);
      w(i,cells+3,6) = -w(i,cells,6);

    }
   }

   return w;

}

double Solver::compute_time_step() {


// Compute the time step for each iteration

  MatrixXd a;
  a.resize(cells+4,cells+4);
  //MatrixXd abs_v = a;


  for (int i=0 ; i < cells+4; i++){
    for(int j=0; j<cells+4; j++){

    double velocity_x= v(i,j,1)/v(i,j,0);
    double velocity_y = v(i,j,3)/v(i,j,0);
    double velocity_z = v(i,j,4)/v(i,j,0);

    double Bsqrd = pow(v(i,j,5),2) + pow(v(i,j,6),2) + pow(v(i,j,7),2);


    double velocity_mag = std::sqrt(pow(velocity_y,2)+pow(velocity_x,2)+pow(velocity_z,2));

    double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2));
    double pressure = (v(i,j,2)-kinetic_energy - 0.5*Bsqrd)*(gamma-1); 

    double cs = std::sqrt((gamma * pressure / v(i,j,0)));
    double ca = std::sqrt(Bsqrd)/std::sqrt(v(i,j,0));
    double cf = std::sqrt(0.5*(pow(cs,2)+pow(ca,2)+std::sqrt(pow(pow(cs,2)+pow(ca,2),2)-4*(pow(cs,2)*pow(v(i,j,5),2)/v(i,j,0)))));
    a(i,j) =  velocity_mag + cf;
    }
  }


  double max_a = a.maxCoeff();
  double time_step = (c * dx) / fabs(max_a); //CFL condition must be half
  return time_step;
}

Tensor<double,3> Solver::euler_flux_y(Tensor<double,3 >v){

// Compute the numerical flux of each solution  
 for (int i =0 ; i< cells+3;i++){
  for(int j=0; j<cells+3;j++){
  
     double velocity_x= (v(i,j,1)/v(i,j,0));
     double velocity_y = v(i,j,3)/v(i,j,0);
     double velocity_z = v(i,j,4)/v(i,j,0);



     double Bsqrd = pow(v(i,j,5),2) + pow(v(i,j,6),2) + pow(v(i,j,7),2);


     double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2)+pow(velocity_z,2));
     double pressure = (v(i,j,2)-kinetic_energy-0.5*Bsqrd)*(gamma-1); 

     flux_initial(i,j,0) = v(i,j,3);// density flux
                                         
     flux_initial(i,j,1) = v(i,j,1)*velocity_y - v(i,j,5)*v(i,j,6); //velocity flux 

     flux_initial(i,j,2) = (v(i,j,2)+pressure+0.5*Bsqrd)*velocity_y - (velocity_x*v(i,j,5) + velocity_y*v(i,j,6) + velocity_z*v(i,j,7))*v(i,j,6);

     flux_initial(i,j,3) = v(i,j,3)*velocity_y + pressure + 0.5*Bsqrd - pow(v(i,j,6),2);

     flux_initial(i,j,4) = (v(i,j,3)*velocity_z); //- (v(i,j,6)*v(i,j,7));

     flux_initial(i,j,5) = v(i,j,5)*velocity_y - v(i,j,6)*velocity_x;

     flux_initial(i,j,6) = 0.;

     flux_initial(i,j,7) = v(i,j,7)*velocity_x - v(i,j,5)*velocity_z;

    } 
  }
  return flux_initial;
}

Tensor<double,3> Solver::euler_flux_x(Tensor<double,3> v){

// Compute the numerical flux of each solution  
for (int i =0 ; i< cells+3;i++){
  for(int j=0; j<cells+3;j++){

    double velocity_x = (v(i,j,1)/v(i,j,0));
    double velocity_y = v(i,j,3)/v(i,j,0);
    double velocity_z = v(i,j,4)/v(i,j,0);

    double Bsqrd = pow(v(i,j,5),2) + pow(v(i,j,6),2) + pow(v(i,j,7),2);

    double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2));
    double pressure = (v(i,j,2)-kinetic_energy-0.5*Bsqrd)*(gamma-1); 

    flux_initial(i,j,0) = v(i,j,1);// density flux
                                         
    flux_initial(i,j,1) = v(i,j,1)*velocity_x + pressure + 0.5*Bsqrd - pow(v(i,j,5),2); //velocity flux 

    flux_initial(i,j,2) = (v(i,j,2)+pressure + 0.5*Bsqrd)*velocity_x - (velocity_x*v(i,j,5) + velocity_y*v(i,j,6) + velocity_z*v(i,j,7))*v(i,j,5);

    flux_initial(i,j,3) = (v(i,j,1))*velocity_y - v(i,j,5)*v(i,j,6);

    flux_initial(i,j,4) = (v(i,j,1)*velocity_z);// - v(i,j,5)*v(i,j,7);

    flux_initial(i,j,5) = 0.;

    flux_initial(i,j,6) = v(i,j,6)*velocity_x - v(i,j,5)*velocity_y;

    flux_initial(i,j,7) = v(i,j,7)*velocity_x - v(i,j,5)*velocity_z;

    
  }
}
return flux_initial;
}

Tensor<double,3> Solver::flux_all(Tensor<double,3> v,char option){

    if(option=='x'){

      for(int k=0; k < 8; k++){
        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

        delta_downwind(i,j,k) = v(i+1,j,k)-v(i,j,k);
          
          }
        }

      for(int i =1;i<cells+3;i++){
        for(int j=1; j<cells+3;j++){
        delta_upwind(i,j,k) = v(i,j,k)-v(i-1,j,k);
        }
      }

      for(int i =1;i<cells+3;i++){
        for(int j=1; j<cells+3;j++){

        delta(i,j,k) = 0.5*delta_upwind(i,j,k)+0.5*delta_downwind(i,j,k);
        r(i,j,k) = delta_upwind(i,j,k) / delta_downwind(i,j,k);

         if(r(i,j,k) <= 0){

          epsalon(i,j,k) = 0;

         }else if(r(i,j,k)>0 && r(i,j,k) <= 1){

          epsalon(i,j,k) = r(i,j,k);

         }else{

          epsalon(i,j,k) = std::min(1.,(2./(1+r(i,j,k))));

         }





        v_L(i,j,k) = v(i,j,k) - 0.5*delta(i,j,k)*epsalon(i,j,k);

        v_R(i,j,k) = v(i,j,k) + 0.5*delta(i,j,k)*epsalon(i,j,k);
        }
      }
    }


    v_L_flux = euler_flux_x(v_L);
    v_R_flux = euler_flux_x(v_R);




    for(int k=0; k < 8; k++){
        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_x(v_L_half);
    v_R_flux = euler_flux_x(v_R_half);

      
 for(int k=0; k < 8; k++){
        for(int i =1;i<cells+2;i++){
          for(int j=1; j<cells+2;j++){
            
        half_time_step(i,j,k) = 0.5*(v_R_half(i,j,k) + v_L_half(i+1,j,k)) - 0.5*(dt/dx)*(-v_R_flux(i,j,k)+v_L_flux(i+1,j,k));

        llf(i,j,k)  = (0.5 * (dx/dt) * (v_R_half(i,j,k)-v_L_half(i+1,j,k))) + (0.5*(v_R_flux(i,j,k) + v_L_flux(i+1,j,k)));
        }
      }
    }


    flux_initial = euler_flux_x(half_time_step);


for(int k=0; k < 8; k++){
        for(int i =1;i<cells+2;i++){
          for(int j=1; j<cells+2;j++){
        rychtmer(i,j,k) = flux_initial(i,j,k);
        }
      }
    }
  flux = 0.5*(rychtmer+llf);

}else{

       for(int k=0; k < 8; k++){

        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){
            delta_downwind(i,j,k) = v(i,j+1,k)-v(i,j,k);
          }
        }

        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

            delta_upwind(i,j,k) = v(i,j,k)-v(i,j-1,k);
          }
        }

        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

            delta(i,j,k) = 0.5*delta_upwind(i,j,k)+0.5*delta_downwind(i,j,k);

            r(i,j,k) = delta_upwind(i,j,k) / delta_downwind(i,j,k);

             if(r(i,j,k) <= 0){

             epsalon(i,j,k) = 0;

             }else if(r(i,j,k)>0 && r(i,j,k) <= 1){

             epsalon(i,j,k) = r(i,j,k);

             }else{

             epsalon(i,j,k) = std::min(1.,(2./(1+r(i,j,k))));

             }




        v_L(i,j,k) = v(i,j,k) - 0.5*delta(i,j,k)*epsalon(i,j,k);

        v_R(i,j,k) = v(i,j,k) + 0.5*delta(i,j,k)*epsalon(i,j,k);
        }
      }
    }

    v_L_flux = euler_flux_y(v_L);
    v_R_flux = euler_flux_y(v_R);


    for(int k=0; k < 8; k++){
        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_y(v_L_half);
    v_R_flux = euler_flux_y(v_R_half);
      
 for(int k=0; k < 8; k++){
        for(int i =1;i<cells+2;i++){
          for(int j=1; j<cells+2;j++){
        half_time_step(i,j,k) = 0.5*(v_R_half(i,j,k) + v_L_half(i,j+1,k)) - 0.5*(dt/dy)*(-v_R_flux(i,j,k)+v_L_flux(i,j+1,k));

        llf(i,j,k)  = (0.5 * (dy/dt) * (v_R_half(i,j,k)-v_L_half(i,j+1,k))) + (0.5*(v_R_flux(i,j,k) + v_L_flux(i,j+1,k)));
        }
      }
    }
flux_initial = euler_flux_y(half_time_step);

  for(int k=0; k < 8; k++){
        for(int i =1;i<cells+2;i++){
          for(int j=1; j<cells+2;j++){
        rychtmer(i,j,k) = flux_initial(i,j,k);
        }
      }
    }

  flux = 0.5*(rychtmer+llf);
  }



return flux;
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
 
  //initialise_input_explosion();
  //initialise_Sod();
  initialise_MHD();
  //initialise_input();
  //initialise_KHI();

  double t = 0;


  do{

  //v = W_to_U(periodic_boundary(w));
  //v = W_to_U(KHI_boundary(w));

  v = W_to_U(periodic_boundary(w));
  std::cout << w;



   dt = compute_time_step();

   t = t+dt;


     flux = flux_all(v,'x');

     flux_initial = euler_flux_x(v);


     for (int i=2; i<cells+2;i++){
      for (int j = 2 ; j<cells+2;j++){
        for (int k =0 ; k<8;k++){

          v_new(i,j,k) = v(i,j,k) - (dt/dx) * (flux(i,j,k)-flux(i-1,j,k));


        }
      }
     }

   w = U_to_W(v_new);

//v_new = periodic_boundary(v_new);
//v_new = W_to_U(periodic_boundary(w));

v_new = W_to_U(periodic_boundary(w));

   flux = flux_all(v_new,'y');
   flux_initial = euler_flux_y(v_new);



   for (int i=2; i<cells+2;i++){
     for (int j = 2 ; j<cells+2;j++){
       for (int k =0 ; k<8;k++){
           v_new(i,j,k) = v_new(i,j,k) - (dt/dy) * (flux(i,j,k)-flux(i,j-1,k));
         }
       }
     }


 v = v_new;

 //sstd::cout << v;
 



  }while (t < final_time);
}

void Solver::save_plot(){

  std::ofstream output("output.dat");

  for(int i = 1; i < cells+1; i++ ){
    for(int j=1; j < cells+1; j++){

    double x = x0 + (i-1)*dx;
    double y = y0 + (j-1)*dy;
    output << x << " " << y << " " << std::sqrt(pow(v(i,j,5),2)+pow(v(i,j,6),2))/v(i,j,7) << " " << "\n";
    output << x << " " << y << " " << v(i,j,0) << " " << "\n";
    }
    output << " " << "\n";
  }
}

int main(){
 Solver solver;
 solver.simulate(); 
 solver.save_plot();
}
