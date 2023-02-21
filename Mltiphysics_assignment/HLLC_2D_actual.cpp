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
    Tensor<double,3> euler_flux(Tensor<double,3> v);
    Tensor<double,3> euler_flux_y(Tensor<double,3> v);
    Tensor<double,3> euler_flux_x(Tensor<double,3> v);
    Tensor<double,3> flux_all(Tensor<double,3> v,char option);
    Tensor<double,3> periodic_boundary(Tensor<double,3 >v);
    Tensor<double,3> transmissive_boundary(Tensor<double,3 >v);
    double bilinear_interpolation(double a, double b,double c,double d);
    MatrixXd ghost_boundary(MatrixXd v);
    void initialise_Sod();
    void save_plot();
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
     Tensor<double,3> star_r;
     Tensor<double,3> star_l;
     MatrixXd level_set;
     MatrixXd level_map;
    double gamma;
};


Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.8;
  cells = 200;
  a = 1.;
  final_time = 0.4;
  x0 = 0.;
  y0 = 0.;
  x1 = 1.;
  y1 = 1.;
  dx = (x1-x0)/cells;
  dy = (y1-y0)/cells;
  gamma = 1.4;
  half_time_step = Tensor<double,3>((cells+4),(cells+4),4);
  flux = Tensor<double,3>((cells+4)+1,(cells+5),4);
  llf = Tensor<double,3>((cells+4)+1,(cells+4),4);
  v_L = Tensor<double,3>((cells+4),(cells+4),4);
  v_R = Tensor<double,3>((cells+4),(cells+4),4);
  v_R_flux = Tensor<double,3>((cells+4),(cells+4),4);
  v_L_flux = Tensor<double,3>((cells+4),(cells+4),4);
  rychtmer = Tensor<double,3>((cells+4),(cells+4),4);
  delta_upwind = Tensor<double,3>((cells+4),(cells+4),4);
  delta_downwind = Tensor<double,3>((cells+4),(cells+4),4);
  delta = Tensor<double,3>((cells+4),(cells+4),4);
  v_L_half = Tensor<double,3>((cells+4),(cells+4),4);
  v_R_half = Tensor<double,3>((cells+4),(cells+4),4);
  r = Tensor<double,3>((cells+4),(cells+4),4);
  epsalon = Tensor<double,3>((cells+4),(cells+4),4);
  half_time_step = Tensor<double,3>((cells+4),(cells+4),4);
  flux_initial = Tensor<double,3>(cells+4,cells+4,4);
  v = Tensor<double,3>(cells+4,cells+4,4); // set up the vector for the solution
  v_new = Tensor<double,3>(cells+4,cells+4,4);
  star_r = Tensor<double,3>(cells+4,cells+4,4);
  star_l = Tensor<double,3>(cells+4,cells+4,4);
  level_set.resize(cells+4,cells+4);
  level_map.resize(cells+4,cells+4);
}


void Solver::initialise_input(){
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



      //level_set(i,j) =  (0.2-std::sqrt(pow((x-0.6),2)+pow((y-0.5),2)));
      level_set(i, j) = -std::max(abs(x - 0.6) - 0.2, abs(y - 0.5) - 0.2);

       //double d1 = sqrt(pow((x - 0.6),2) + pow((y - 0.35),2)) - 0.2;
       //double d2 = sqrt(pow((x - 0.6),2) + pow((y - 0.65),2)) - 0.2;
       //level_set(i,j) = -std::min(d1,d2);

      if(x < 0.2){

        pressure =1.5698;
        velocity_x = 0.394;
        velocity_y= 0.;
        density = 1.3764;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));

      }else if (x>=0.2){

        pressure = 1.;
        velocity_x =0.;
        velocity_y = 0.;
        density = 1.;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));
      }


      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
      input << x << " " << y << " " << level_set(i,j) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
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

 for(int i=0; i < cells+2; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cells+2; j++){

        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;

      if(x < 0.5 && y < 0.5){

        pressure =1.0;
        velocity_x = 0.;
        velocity_y= 0.;
        density = 0.8;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));

      }else if (x>0.5 && y <0.5){

        pressure = 1.;
        velocity_x =0.;
        velocity_y = 0.7276;
        density = 1.;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

      }else if(x < 0.5 && y > 0.5){

        pressure = 1.;
        velocity_x =0.7276;
        velocity_y = 0.;
        density = 1.;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

      }else if(x > 0.5 && y > 0.5){

        pressure = 0.4;
        velocity_x =0.;
        velocity_y = 0.;
        density = 0.5313;
        energy = pressure/(gamma-1) + 0.5*density*(pow(velocity_x,2)+pow(velocity_y,2));

      }

      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
      input << x << " " << y << " " << v(i,j,0) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
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

        
        level_set(i,j) = 0.2 - std::sqrt(pow((x-0.6),2)+pow((y-0.5),2));


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

        input << x << " " << y << " " << v(i,j,1) << " "  << level_set(i,j) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";

      }
    }

}

Tensor<double,3> Solver::periodic_boundary(Tensor<double,3 >v){

     for(int k=0; k<4; k++){
    for(int j=0; j<cells+4; j++){
      v(0,j,k) = v(cells,j,k);
      v(1,j,k) = v(cells+1,j,k);

      v(cells+2,j,k) = v(2,j,k);
      v(cells+3,j,k) = v(3,j,k);

    }


    for(int i=0; i<cells+4; i++){
      v(i,0,k) = v(i,cells,k);
      v(i,1,k) = v(i,cells+1,k);

      v(i,cells+2,k) = v(i,2,k);
      v(i,cells+3,k) = v(i,3,k);
    }
   }
return v;
}

Tensor<double,3> Solver::transmissive_boundary(Tensor<double,3 >v){

      for(int k=0; k<4; k++){
    for(int j=0; j<cells+4; j++){
      v(0,j,k) = v(2,j,k);
      v(1,j,k) = v(2,j,k);

      v(cells+2,j,k) = v(cells+1,j,k);
      v(cells+3,j,k) = v(cells+1,j,k);

    }


    for(int i=0; i<cells+4; i++){
      v(i,0,k) = v(i,2,k);
      v(i,1,k) = v(i,2,k);

      v(i,cells+2,k) = v(i,cells+1,k);
      v(i,cells+3,k) = v(i,cells+1,k);
    }
   }
return v;
}

MatrixXd Solver::ghost_boundary(MatrixXd k){

  double normal_x,normal_y;
  VectorXf u_fluid;
  u_fluid.resize(4);
  std::ofstream map("level_set.dat");
  VectorXf u_rigid;


  for (int i =2 ; i<cells+3;i++){
    for(int j=2;j<cells+3;j++){

// this is the sana logic

      if(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0){ 
        //std::cout << (level_set(i,j)*level_set(i-1,j) < 0)  << " OR " <<  (level_set(i,j)*level_set(i,j-1)) << std::endl;
        double x = x0 + (i-0.5)*dx;
        double y = y0 + (j-0.5)*dy;

        normal_x = (level_set(i+1,j) - level_set(i-1,j))/(2*dx), normal_y = (level_set(i,j+1) - level_set(i,j-1))/(2*dy);

        double xp_i = x - normal_x*level_set(i,j) , xp_j = y - normal_y*level_set(i,j); 
        double point_1_x =  xp_i + 1.5*dx*normal_x, point_1_y = xp_j + 1.5*dy*normal_y; // coords of 1st point 
        double point_2_x =  xp_i - 1.5*dx*normal_x, point_2_y = xp_j - 1.5*dy*normal_y; // coords of 2nd point 


        int p_i = round((point_1_x - x0)/dx + 0.5);
        int p_j = round((point_1_y - y0)/dy + 0.5 );

        int p_i_2 = round((point_2_x - x0)/dx + 0.5 );
        int p_j_2 = round((point_2_y - y0)/dy + 0.5);

        double x_centre = x0 + (p_i_2-0.5)*dx;
        double y_centre = y0 + (p_j_2-0.5)*dy;

    for (int k =0;k<4;k++){
        if (point_2_x > x_centre && point_2_y > y_centre){
          //std::cout << v(p_i,p_j,0) << " " << v(p_i+1,p_j+1,0)  << " " << v(p_i+1,p_j,0) << " " << v(p_i,p_j+1,0) << " "<< "tr" << " " <<  "\n";


          double interpolated_x = (((x_centre+dx)-point_2_x)*v(p_i_2,p_j_2+1,k)/dx) + ((point_2_x-x_centre)*v(p_i_2+1,p_j_2+1,k)/dx);
          double interpolated_y = ((x_centre+dx)-point_2_x)*v(p_i_2,p_j_2,k)/dx + (point_2_x-x_centre)*v(p_i_2+1,p_j_2,k)/dx;
          

          double interpolated = ((y_centre+dy)-point_2_y)*interpolated_y/dy + (point_2_y-(y_centre))*interpolated_x/dy;
                    //std::cout << interpolated << "\n";
          u_fluid(k) = interpolated;

          //std::cout << interpolated << "\n";



        }else if (point_2_x > x_centre && point_2_y < y_centre){
         

          double interpolated_x = ((x_centre+dx)-point_2_x)*v(p_i_2,p_j_2,k)/dx + (point_2_x-x_centre)*v(p_i_2+1,p_j_2,k)/dx;
          double interpolated_y = ((x_centre+dx)-point_2_x)*v(p_i_2,p_j_2-1,k)/dx + (point_2_x-x_centre)*v(p_i_2+1,p_j_2-1,k)/dx;

          double interpolated = (y_centre-point_2_y)*interpolated_y/dy + (point_2_y-(y_centre-dy))*interpolated_x/dy; 

          u_fluid(k) = interpolated;

          //std::cout << interpolated << "\n";
        }else if (point_2_x < x_centre && point_2_y < y_centre){
          

          double interpolated_x = ((x_centre)-point_2_x)*v(p_i_2-1,p_j_2,k)/dx + (point_2_x-(x_centre-dx))*v(p_i_2,p_j_2,k)/dx;
          double interpolated_y = ((x_centre)-point_2_x)*v(p_i_2-1,p_j_2-1,k)/dx + (point_2_x-(x_centre-dx))*v(p_i_2,p_j_2-1,k)/dx;

          double interpolated = ((y_centre)-point_2_y)*interpolated_y/dy + (point_2_y-(y_centre-dy))*interpolated_x/dy; 

          u_fluid(k) = interpolated;
                    //std::cout << interpolated << "\n";


        }else if (point_2_x < x_centre && point_2_y > y_centre){

          double interpolated_x = ((x_centre)-point_2_x)*v(p_i_2-1,p_j_2+1,k)/dx + (point_2_x-(x_centre-dx))*v(p_i_2,p_j_2+1,k)/dx;
          double interpolated_y = ((x_centre)-point_2_x)*v(p_i_2-1,p_j_2,k)/dx + (point_2_x-(x_centre-dx))*v(p_i_2,p_j_2,k)/dx;

          double interpolated = ((y_centre+dy)-point_2_y)*interpolated_y/dy + (point_2_y-(y_centre))*interpolated_x/dy;
          u_fluid(k) = interpolated; 

//uses interafcial cells
          //std::cout << interpolated << "\n";



        }

}

       double normal_velocity_real = (u_fluid(1)/u_fluid(0))*normal_x + (u_fluid(3)/u_fluid(0))*normal_y;
       double normal_velocity_rigid = -normal_velocity_real;
       double tangential_velocity_x_real = (u_fluid(1)/u_fluid(0)) - normal_velocity_real*normal_x , tangential_velocity_y_real = (u_fluid(3)/u_fluid(0)) - normal_velocity_real*normal_y;
// HLLC  Right is the real liquid left is the rigid body

      double total_velocity_r = pow((u_fluid(1)/u_fluid(0)),2) + pow(u_fluid(3)/u_fluid(0),2);
      double pressure_r = ((u_fluid(2)-(0.5*u_fluid(0)*total_velocity_r))*(gamma-1));
      double cs_r = std::sqrt((gamma * (pressure_r)) /u_fluid(0));

      double total_velocity_l = pow((u_fluid(1)/u_fluid(0)),2) + pow(u_fluid(3)/u_fluid(0),2);
      double pressure_l = ((u_fluid(2)-(0.5*u_fluid(0)*total_velocity_r))*(gamma-1));
      double cs_l = std::sqrt((gamma * (pressure_r)) /u_fluid(0));

      double s_r = std::min(abs(normal_velocity_real) + cs_r,abs(normal_velocity_rigid)+cs_l);
      double s_l = -s_r;

      double v_star = pressure_r - pressure_l + (u_fluid(0)*normal_velocity_real*(s_l-normal_velocity_real)) - (u_fluid(0)*normal_velocity_rigid*(s_r-normal_velocity_rigid))/(u_fluid(0)*(s_l-normal_velocity_real) - u_fluid(0)*(s_r-normal_velocity_rigid));
    
      double density_star = u_fluid(0)*((s_r - normal_velocity_rigid)/(s_r - v_star));
      double density_velocity_star = density_star*v_star;
      double energy_star = density_star * ((u_fluid(2))/u_fluid(0)) + (v_star-normal_velocity_rigid)*(v_star + ((pressure_r)/(u_fluid(0)*(s_r-normal_velocity_rigid))));
      double velocity_x_star = (density_velocity_star/density_star)*normal_x + tangential_velocity_x_real, velocity_y_star = (density_velocity_star/density_star)*normal_y + tangential_velocity_y_real;


      u_fluid(0) = density_star;
      u_fluid(1) = density_star*velocity_x_star;
      u_fluid(2) = energy_star;
      u_fluid(3) = density_star*velocity_y_star;

      v(i,j,0) = u_fluid(0);
      v(i,j,1) = u_fluid(1);
      v(i,j,2) = u_fluid(2);
      v(i,j,3) = u_fluid(3); // set the values in the interfacial cells in the domain
      //level_map(i,j) = 2;
      //std::cout << u_fluid(1) << "\n";

      }
    }
  }

for (int i =2 ; i<cells+3;i++){
    for(int j=2;j<cells+3;j++){



      if (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0){ 
          level_map(i,j) = pow(10,100);
          v(i,j,0) = pow(10,100);
          v(i,j,1) = pow(10,100);
          v(i,j,2) = pow(10,100);
          v(i,j,3) = pow(10,100);
          //v(i,j,4) = pow(10,100);

      }
    }
  }


// sweeping logic


   for (int i =2 ; i<cells+3;i++){
     for(int j=2;j<cells+3;j++){

      // This logic basically selects everything that is NOT an interface and that its level set function is bigger than 0 AND it requires that the level set function is increasing, it might not work with level set functions that have negative values inside the rigid body



        if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i+1,j) > level_set(i,j))){

         double  normal_x = abs(level_set(i+1,j) - level_set(i-1,j)/(2*dx)), normal_y = abs(level_set(i,j+1) - level_set(i,j-1)/(2*dy));

        for(int k = 0; k<4; k++ ){
         double Qx = std::min(v(i-1,j,k),v(i+1,j,k));
         double Qy = std::min(v(i,j-1,k),v(i,j+1,k));

         v(i,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
         //std::cout << v(i,j,3) << "\n";
         level_map(i,j) = 1;

         if(level_set(i+2,j) <= level_set(i+1,j)){

            v(i+1,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
            //level_map(i+1,j) = 1; 
         }

         
        }
       }

     }
   }


      for (int i =2 ; i<cells+3;i++){
     for(int j=2;j<cells+3;j++){

        if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i,j+1) > level_set(i,j))){

         double  normal_x = abs(level_set(i+1,j) - level_set(i-1,j)/(2*dx)), normal_y = abs(level_set(i,j+1) - level_set(i,j-1)/(2*dy));

        for(int k = 0; k<4; k++ ){
         double Qx = std::min(v(i-1,j,k),v(i+1,j,k));
         double Qy = std::min(v(i,j-1,k),v(i,j+1,k));

         v(i,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
         level_map(i,j) = 1;

         if(level_set(i,j+2) <= level_set(i,j+1)){

         v(i,j+1,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
         //level_map(i,j+1) = 1;
         }

         //std::cout << v(i,j,k)<<"\n";
        }

       }
       }
       }




    for (int i =cells+2 ; i>2;i--){
      for(int j=2;j<cells+3;j++){

         if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i-1,j) > level_set(i,j))){

          double  normal_x = abs(level_set(i+1,j) - level_set(i-1,j)/(2*dx)), normal_y = abs(level_set(i,j+1) - level_set(i,j-1)/(2*dy));

         for(int k = 0; k<4; k++ ){
          double Qx = std::min(v(i-1,j,k),v(i+1,j,k));
          double Qy = std::min(v(i,j-1,k),v(i,j+1,k));
          level_map(i,j) = 1;

          v(i,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));


          if(level_set(i-2,j) <= level_set(i-1,j)){

               v(i-1,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
               //level_map(i-1,j) = 1;
            }

         }

        
        }
        }
        }


          for (int i =2 ; i<cells+3;i++){
            for(int j=cells+2;j>2;j--){

         if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i,j-1) > level_set(i,j))){

          double  normal_x = abs(level_set(i+1,j) - level_set(i-1,j)/(2*dx)), normal_y = abs(level_set(i,j+1) - level_set(i,j-1)/(2*dy));

         for(int k = 0; k<4; k++ ){
          double Qx = std::min(v(i-1,j,k),v(i+1,j,k));
          double Qy = std::min(v(i,j-1,k),v(i,j+1,k));

          v(i,j,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
          //std::cout << v(i,j,k)<<"\n";

          if(level_set(i,j-2) <= level_set(i,j-1)){

                v(i,j-1,k) = 1/((normal_x/dx)+(normal_y/dy))* (((normal_x*Qx)/dx) + ((normal_y*Qy)/dy));
                //level_map(i,j-1) = 1;
           }
        }
        }
        }
        }





 std::ofstream output("output.dat");

  for(int i = 0; i < cells+4; i++ ){
    for(int j=0; j < cells+4; j++){

    double x = x0 + (i-0.5)*dx;
    double y = y0 + (j-0.5)*dy;
    output << x << " " << y << " " << v(i,j,2) << " " << "\n";
    }
  }


    for(int i = 0; i < cells+4; i++ ){
    for(int j=0; j < cells+4; j++){

    double x = x0 + (i-0.5)*dx;
    double y = y0 + (j-0.5)*dy;
    map << x << " " << y << " " << level_map(i,j) << " " << "\n";
    }
    map << " " << "\n";
  }

  return level_map;
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
    double velocity_mag = std::sqrt(pow(velocity_y,2)+pow(velocity_x,2));

    double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2));
    double pressure = (v(i,j,2)-kinetic_energy)*(gamma-1); 

    double cs = std::sqrt((gamma * pressure / v(i,j,0)));
    a(i,j) =  velocity_mag + cs;
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

     double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2));
     double pressure = (v(i,j,2)-kinetic_energy)*(gamma-1); 

     flux_initial(i,j,0) = v(i,j,3);// density flux
                                         
     flux_initial(i,j,1) = v(i,j,1)*velocity_y; //velocity flux 

     flux_initial(i,j,2) = (v(i,j,2)+pressure)*velocity_y;

     flux_initial(i,j,3) = v(i,j,3)*velocity_y + pressure;
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
    double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2));
    double pressure = (v(i,j,2)-kinetic_energy)*(gamma-1); 

    flux_initial(i,j,0) = v(i,j,1);// density flux
                                         
    flux_initial(i,j,1) = v(i,j,1)*velocity_x + pressure; //velocity flux 

    flux_initial(i,j,2) = (v(i,j,2)+pressure)*velocity_x;

    flux_initial(i,j,3) = (v(i,j,1))*velocity_y;
    
  }
}
return flux_initial;
}

Tensor<double,3> Solver::flux_all(Tensor<double,3> v,char option){

    if(option=='x'){

      for(int k=0; k < 4; k++){
        for(int i =0;i<cells+3;i++){
          for(int j=0; j<cells+3;j++){

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

    for(int k=0; k < 4; k++){
        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_x(v_L_half);
    v_R_flux = euler_flux_x(v_R_half);

    for (int i = 1 ; i < cells+2; i++){
      for (int j =1 ; j < cells+2 ; j++){


      double velocity_x_l = v_R_half(i,j,1)/v_R_half(i,j,0);
      double velocity_y_l = v_R_half(i,j,3)/v_R_half(i,j,0);

      double total_velocity_l = pow(velocity_x_l,2) + pow(velocity_y_l,2);
      double pressure_l = ((v_R_half(i,j,2)-(0.5*v_R_half(i,j,0)*total_velocity_l))*(gamma-1));
      double cs_l = std::sqrt((gamma * (pressure_l)) /v_R_half(i,j,0));

      double velocity_x_r = v_L_half(i+1,j,1)/v_L_half(i+1,j,0);
      double velocity_y_r = v_L_half(i+1,j,3)/v_L_half(i+1,j,0);
      double total_velocity_r = pow(velocity_x_r,2)+pow(velocity_y_r,2);

      double pressure_r = ((v_L_half(i+1,j,2)-(0.5*v_L_half(i+1,j,0)*total_velocity_r))*(gamma-1));
      double cs_r = std::sqrt((gamma * (pressure_r)) /v_L_half(i+1,j,0));
    
// -----------------------------------------------------------------------------------------------------------------


      double a_bar = 0.5*(cs_r+cs_l);
      double fi_bar = 0.5*(v_L_half(i+1,j,0)+v_R_half(i,j,0));
      double p_pvrs = 0.5*(pressure_l+pressure_r) - 0.5*(velocity_x_r-velocity_x_l)*a_bar*fi_bar;
      double pressure_star = std::max(0.,p_pvrs);
      double qr;
      double ql;

      if(pressure_star <= pressure_r){

        qr = 1;

      }else if (pressure_star > pressure_r){
        qr = pow(1+((gamma+1)/(2*gamma))*(pressure_star/(pressure_r-1)),0.5);
      }


      if(pressure_star <= pressure_l){

        ql = 1;

      }else if (pressure_star > pressure_l){
        ql = pow(1+((gamma+1)/(2*gamma))*(pressure_star/(pressure_l-1)),0.5);
      }

      double s_r = std::max(abs(velocity_x_r) + cs_r,abs(velocity_x_l)+cs_l);
      double s_l = -s_r;

      double v_star = (pressure_r - pressure_l + v_R_half(i,j,1)*(s_l-velocity_x_l) - v_L_half(i+1,j,1)*(s_r-velocity_x_r))/(v_R_half(i,j,0)*(s_l-velocity_x_l) - v_L_half(i+1,j,0)*(s_r-velocity_x_r));
    
      star_r(i,j,0) = v_L_half(i+1,j,0)*((s_r - velocity_x_r)/(s_r - v_star));
      star_r(i,j,1) = star_r(i,j,0)*v_star;
      star_r(i,j,2) = star_r(i,j,0) * ((v_L_half(i+1,j,2))/(v_L_half(i+1,j,0)) + (v_star-velocity_x_r)*(v_star + ((pressure_r)/(v_L_half(i+1,j,0)*(s_r-velocity_x_r)))));
      star_r(i,j,3) = star_r(i,j,0)*velocity_y_r;

      star_l(i,j,0) = v_R_half(i,j,0)*((s_l - velocity_x_l)/(s_l - v_star));
      star_l(i,j,1) = star_l(i,j,0)*v_star;
      star_l(i,j,2) = star_l(i,j,0)*(((v_R_half(i,j,2))/(v_R_half(i,j,0))) + (v_star-velocity_x_l)*(v_star + ((pressure_l)/(v_R_half(i,j,0)*(s_l-velocity_x_l)))));
      star_l(i,j,3) = star_l(i,j,0)*velocity_y_l;




      for(int k=0; k<4;k++){
        if(0<=s_l){

        flux(i,j,k) = v_R_flux(i,j,k);

      }else if (s_l<=0 && 0<=v_star){
        
        flux(i,j,k) = v_R_flux(i,j,k) + s_l*(star_l(i,j,k) - v_R_half(i,j,k));

      }else if(v_star <= 0 && 0<=s_r){

        flux(i,j,k) = v_L_flux(i+1,j,k)+s_r*(star_r(i,j,k)-v_L_half(i+1,j,k));

      }else if(s_r<=0){

        flux(i,j,k) = v_L_flux(i+1,j,k);
    }
    }

    }
    }

}else{

       for(int k=0; k < 4; k++){

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

            delta(i,j,k) = 0.5*delta_upwind(i,j,k) + 0.5*delta_downwind(i,j,k);

            r(i,j,k) = delta_upwind(i,j,k) / delta_downwind(i,j,k);

            if(r(i,j,k) <= 0){

            epsalon(i,j,k) = 0;

            }else if(r(i,j,k)>0 && r(i,j,k) <= 1){

            epsalon(i,j,k) = r(i,j,k);

            }else if(r(i,j,k)>1){

            epsalon(i,j,k) = std::min(1.,(2./(1+r(i,j,k))));

            }

        v_L(i,j,k) = v(i,j,k) - 0.5*delta(i,j,k)*epsalon(i,j,k);

        v_R(i,j,k) = v(i,j,k) + 0.5*delta(i,j,k)*epsalon(i,j,k);
        }
      }
    }

    v_L_flux = euler_flux_y(v_L);
    v_R_flux = euler_flux_y(v_R);

    for(int k=0; k < 4; k++){
        for(int i =1;i<cells+3;i++){
          for(int j=1; j<cells+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_y(v_L_half);
    v_R_flux = euler_flux_y(v_R_half);




    for (int i = 1 ; i < cells+2; i++){
      for (int j =1 ; j < cells+2 ; j++){

      double velocity_x_l = v_R_half(i,j,1)/v_R_half(i,j,0);
      double velocity_y_l = v_R_half(i,j,3)/v_R_half(i,j,0);

      double total_velocity_l = pow(velocity_x_l,2) + pow(velocity_y_l,2);
      double pressure_l = ((v_R_half(i,j,2)-(0.5*v_R_half(i,j,0)*total_velocity_l))*(gamma-1));
      double cs_l = std::sqrt((gamma * (pressure_l)) /v_R_half(i,j,0));

      double velocity_x_r = v_L_half(i,j+1,1)/v_L_half(i,j+1,0);
      double velocity_y_r = v_L_half(i,j+1,3)/v_L_half(i,j+1,0);
      double total_velocity_r = pow(velocity_x_r,2)+pow(velocity_y_r,2);

      double pressure_r = ((v_L_half(i,j+1,2)-(0.5*v_L_half(i,j+1,0)*total_velocity_r))*(gamma-1));
      double cs_r = std::sqrt((gamma * (pressure_r)) /v_L_half(i,j+1,0));
    
// -----------------------------------------------------------------------------------------------------------------


      double a_bar = 0.5*(cs_r+cs_l);
      double fi_bar = 0.5*(v_L_half(i,j+1,0)+v_R_half(i,j,0));
      double p_pvrs = 0.5*(pressure_l+pressure_r) - 0.5*(velocity_y_r-velocity_y_l)*a_bar*fi_bar;
      double pressure_star = std::max(0.,p_pvrs);
      double qr;
      double ql;

      if(pressure_star <= pressure_r){

        qr = 1;

      }else if(pressure_star > pressure_r) {
        qr = pow(1+(((gamma+1)/(2*gamma))*(pressure_star/(pressure_r-1))),0.5);
      }


      if(pressure_star <= pressure_l){

        ql = 1;

      }else if (pressure_star > pressure_l){

        ql = pow(1+((gamma+1)/(2*gamma))*(pressure_star/(pressure_l-1)),0.5);

      }

      double s_r = std::min(abs(velocity_y_r) + cs_r,abs(velocity_y_l)+cs_l);
      double s_l = -s_r;


      double v_star = (pressure_r - pressure_l + v_R_half(i,j,3)*(s_l-velocity_y_l) - v_L_half(i,j+1,3)*(s_r-velocity_y_r))/(v_R_half(i,j,0)*(s_l-velocity_y_l) - v_L_half(i,j+1,0)*(s_r-velocity_y_r));
    
      star_r(i,j,0) = v_L_half(i,j+1,0)*((s_r - velocity_y_r)/(s_r - v_star));
      star_r(i,j,1) = star_r(i,j,0)*velocity_x_r;
      star_r(i,j,2) = star_r(i,j,0) * ((v_L_half(i,j+1,2))/(v_L_half(i,j+1,0)) + (v_star-velocity_y_r)*(v_star + ((pressure_r)/(v_L_half(i,j+1,0)*(s_r-velocity_y_r)))));
      star_r(i,j,3) = star_r(i,j,0)*v_star;

      star_l(i,j,0) = v_R_half(i,j,0)*((s_l - velocity_y_l)/(s_l - v_star));
      star_l(i,j,1) = star_l(i,j,0)*velocity_x_l;
      star_l(i,j,2) = star_l(i,j,0)*(((v_R_half(i,j,2))/(v_R_half(i,j,0))) + (v_star-velocity_y_l)*(v_star + ((pressure_l)/(v_R_half(i,j,0)*(s_l-velocity_y_l)))));
      star_l(i,j,3) = star_l(i,j,0)*v_star;




      for(int k=0; k<4;k++){
        if(0<=s_l){

        flux(i,j,k) = v_R_flux(i,j,k);

      }else if (s_l<=0 && 0<=v_star){
        
        flux(i,j,k) = v_R_flux(i,j,k) + s_l*(star_l(i,j,k) - v_R_half(i,j,k));

      }else if(v_star <= 0 && 0<=s_r){

        flux(i,j,k) = v_L_flux(i,j+1,k)+s_r*(star_r(i,j,k)-v_L_half(i,j+1,k));

      }else if(s_r<=0){

        flux(i,j,k) = v_L_flux(i,j+1,k);
    }
    }
    }
    }
      
  }
//std :: cout << flux;
return flux;
}

void Solver::simulate(){
 
  //initialise_input_explosion();
  //initialise_Sod();
  initialise_input();
  //ghost_boundary(level_set);
  ghost_boundary(level_set);
  exit(0);
  std::ofstream output_animation("output_animation.dat");
  
  
  double t = 0;


  do{
   ghost_boundary(level_set);
   v = transmissive_boundary(v);

   dt = compute_time_step();
   std::cout << dt << std::endl;
   t = t+dt;


     flux = flux_all(v,'x');
     flux_initial = euler_flux_x(v);
   
     for (int i=2; i<cells+2;i++){
      for (int j = 2 ; j<cells+2;j++){
        for (int k =0 ; k<4;k++){
          v_new(i,j,k) = v(i,j,k) - (dt/dx) * (flux(i,j,k)-flux(i-1,j,k));
        }
      }
     }

      v_new = periodic_boundary(v_new);

      flux = flux_all(v_new,'y');
      flux_initial = euler_flux_y(v_new);

      for (int i=2; i<cells+2;i++){
        for (int j = 2 ; j<cells+2;j++){

              double x = x0 + (i-0.5)*dx;
              double y = y0 + (j-0.5)*dy;

          for (int k =0 ; k<4;k++){
              v_new(i,j,k) = v_new(i,j,k) - (dt/dy) * (flux(i,j,k)-flux(i,j-1,k));              
            }

          output_animation << t << " " << x << " " << y << " " << v_new(i,j,0) << " " <<" " << level_map(i,j)<< "\n";

          }
          output_animation << " " << "\n";
        }
        output_animation << "\n";
        output_animation << "\n";


  v = v_new;





  }while (t < final_time);
}

void Solver::save_plot(){

  std::ofstream output("output.dat");

  for(int i = 2; i < cells+2; i++ ){
    for(int j=2; j < cells+2; j++){

    double x = x0 + (i-1)*dx;
    double y = y0 + (j-1)*dy;
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
