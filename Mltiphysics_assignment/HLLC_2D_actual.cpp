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
    MatrixXd  evolve_level_set(MatrixXd set,double vx,double vy);
    void initialise_input_explosion();
    void moving_rigid_body();
    void moving_rigid_body_moving();
    void rotating_rigid_body();
    Tensor<double,3> euler_flux(Tensor<double,3> v);
    Tensor<double,3> euler_flux_y(Tensor<double,3> v);
    Tensor<double,3> euler_flux_x(Tensor<double,3> v);
    Tensor<double,3> flux_all(Tensor<double,3> v,char option);
    Tensor<double,3> periodic_boundary(Tensor<double,3 >v);
    Tensor<double,3> transmissive_boundary(Tensor<double,3 >v);
    void velocity_field(double time);
    double bilinear_interpolation(double a, double b,double c,double d);
    void ghost_boundary();
    void level_set_reinitialisation();
    void initialise_Sod();
    void save_plot();
    void SLIC();
    MatrixXd advect_level_set(MatrixXd set,double vx,double vy,double time);

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
    double loops;
    int cells;
    int cellsx;
    int cellsy;
    double pi;
    double ang_vel;
    double v_x_rot;
    double v_y_rot;
    double t;
    //double t;
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
     Tensor<double,3> v_temp;
     MatrixXd level_set;
     MatrixXd level_set_new;
     MatrixXd level_map;
    double gamma;
};


Solver::Solver(){
  std::cout << "Constructing Solver.." << "\n";
  c = 0.8;
  loops=0;
  //cells = 200;
  cellsx = 100;
  cellsy = 100;
  a = 1.;
  final_time = 0.4;
  x0 = 0.;
  y0 = 0.;
  x1 = 1.;
  y1 = 1.;
  dx = (x1-x0)/cellsx;
  dy = (y1-y0)/cellsy;
  t=0;
  pi=2*std::acos(0.0);
  gamma = 1.4;
  ang_vel = 2*pi/final_time;
  half_time_step = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  flux = Tensor<double,3>((cellsx+4)+1,(cellsy+5),4);
  llf = Tensor<double,3>((cellsx+4)+1,(cellsy+4),4);
  v_L = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  v_R = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  v_R_flux = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  v_L_flux = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  rychtmer = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  delta_upwind = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  delta_downwind = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  delta = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  v_L_half = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  v_R_half = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  r = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  epsalon = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  half_time_step = Tensor<double,3>((cellsx+4),(cellsy+4),4);
  flux_initial = Tensor<double,3>(cellsx+4,cellsy+4,4);
  v = Tensor<double,3>(cellsx+4,cellsy+4,4); // set up the vector for the solution
  v_new = Tensor<double,3>(cellsx+4,cellsy+4,4);
  star_r = Tensor<double,3>(cellsx+4,cellsy+4,4);
  star_l = Tensor<double,3>(cellsx+4,cellsy+4,4);
  level_set.resize(cellsx+4,cellsy+4);
  level_set_new.resize(cellsx+4,cellsy+4);
  level_map.resize(cellsx+4,cellsy+4);
   v_temp = Tensor<double,3>(cellsx+4,cellsy+4,4);
}


void Solver::initialise_input(){
  std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cellsx+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cellsy+4; j++){

      double x = x0 + (i-0.5)*dx;
      double y = y0 + (j-0.5)*dy;


// circle 
      //level_set(i,j) =  (0.2-std::sqrt(pow((x-0.6),2)+pow((y-0.5),2)));

//square - NOT WORKING
      //level_set(i, j) = -std::max(abs(x - 0.6) - 0.2, abs(y - 0.5) - 0.2)

      if(x <0.6 && y >= x-0.1 && y <= 1.1-x){
        level_set(i,j) = (x-0.4);
      }else if (x > 0.6 && y <= x-0.1 && y >= 1.1-x){
        level_set(i,j) = (0.8-x);
      }else if(y > 0.5 && x < y+0.1 && x > 1.1-y){
        level_set(i,j) = (0.7-y);
      }else if ( y < 0.5 && x > y+0.1 && x < 1.1-y){
        level_set(i,j)=(y-0.3);
      }

// overlapping circles

        //double d1 = sqrt(pow((x - 0.6),2) + pow((y - 0.35),2)) - 0.2;
        //double d2 = sqrt(pow((x - 0.6),2) + pow((y - 0.65),2)) - 0.2;
        //level_set(i,j) = -std::min(d1,d2);

 //non overlapping circles

       //double d1 = sqrt(pow((x - 0.6),2) + pow((y - 0.25),2)) - 0.2;
       //double d2 = sqrt(pow((x - 0.6),2) + pow((y - 0.75),2)) - 0.2;
       //level_set(i,j) = -std::min(d1,d2);


  // square 


    // const double l = 0.4;
    // const double cx = 0.6;
    // const double cy = 0.5;
    
    // // Calculate distances to sides of square
    // double d_top = y - (cy + l/2);
    // double d_bottom = (cy - l/2) - y;
    // double d_left = x - (cx - l/2);
    // double d_right = (cx + l/2) - x;
    
    // // Calculate distances to corners approximated as circular arcs
    // double d_top_left = sqrt(pow(x - cx + l/2, 2) + pow(y - cy + l/2, 2)) - l/2;
    // double d_top_right = sqrt(pow(x - cx - l/2, 2) + pow(y - cy + l/2, 2)) - l/2;
    // double d_bottom_left = sqrt(pow(x - cx + l/2, 2) + pow(y - cy - l/2, 2)) - l/2;
    // double d_bottom_right = sqrt(pow(x - cx - l/2, 2) + pow(y - cy - l/2, 2)) - l/2;
    
    // // Find minimum distance and return signed distance
    // double dist = std::min({d_top, d_bottom, d_left, d_right, d_top_left, d_top_right, d_bottom_left, d_bottom_right});
    // level_set(i,j) = (x >= cx - l/2 && x <= cx + l/2 && y >= cy - l/2 && y <= cy + l/2) ? dist : -dist;

       //level_set(i,j) = std::min(0.2-std::sqrt(pow((x-0.6),2) + pow(y-0.25,2)),0.2 - std::sqrt(pow((x-0.6),2) + pow(y-0.75,2)));

      if( x < 0.2){

        pressure =1.5698;
        velocity_x = 0.394;
        velocity_y= 0.;
        density = 1.3764;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));

      }else if (x >=0.2){

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

void Solver::moving_rigid_body(){

  std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cellsx+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cellsy+4; j++){

      double x = x0 + (i-0.5)*dx;
      double y = y0 + (j-0.5)*dy;


// circle 
      level_set(i,j) =  (0.2-std::sqrt(pow((x-0.5),2)+pow((y-0.5),2)));

    

        pressure =1.0;
        velocity_x = 1.;
        velocity_y= 0.;
        density = 1.;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));


      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
      input << x << " " << y << " " << level_set(i,j) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }

}

void Solver::moving_rigid_body_moving(){

std::ofstream input("input.dat");

   double pressure;
   double velocity_x;
   double velocity_y;
   double density;
   double energy;


// set up the initial values of the solution and put them into a file called input.

  for(int i=0; i < cellsx+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cellsy+4; j++){

      double x = x0 + (i-0.5)*dx;
      double y = y0 + (j-0.5)*dy;


// circle 
      level_set(i,j) =  (0.2-std::sqrt(pow((x-1.5),2)+pow((y-0.5),2)));

      //double d1 = sqrt(pow((x - 0.6),2) + pow((y - 0.35),2)) - 0.2;
      //double d2 = sqrt(pow((x - 0.6),2) + pow((y - 0.65),2)) - 0.2;
      //level_set(i,j) = -std::min(d1,d2);

    

        pressure =1.0;
        velocity_x = 0.;
        velocity_y= 0.;
        density = 1.;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));


      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
      input << x << " " << y << " " << level_set(i,j) << " " << "\n";  // << " " //<< v(i,j,1)/v(i,j,0) << " " <<  (v(i,j,2)-(0.5*v(i,j,0)*pow((v(i,j,1)/v(i,j,0)),2)))*(gamma-1) << "\n";
    }
  }



}

void Solver::rotating_rigid_body(){

   double pressure;
   double velocity_x;
   double velocity_y;
   double density;
   double energy;


  for(int i=0; i < cellsx+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cellsy+4; j++){

      double x = x0 + (i-0.5)*dx;
      double y = y0 + (j-0.5)*dy;


// circle 
      level_set(i,j) =  (0.15-std::sqrt(pow((x-0.2),2)+pow((y-0.5),2)));

    

        pressure =1.0;
        velocity_x = 0.;
        velocity_y= 0.;
        density = 1.;
        energy = (pressure/(gamma-1)) + (0.5*density*(pow(velocity_x,2)+pow(velocity_y,2)));


      v(i,j,0) = density;
      v(i,j,1) = velocity_x * density; // the momentum value 
      v(i,j,2) = energy;
      v(i,j,3) = velocity_y*density;
    
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
    for(int j=0; j<cellsy+4; j++){
      v(0,j,k) = v(cellsx,j,k);
      v(1,j,k) = v(cellsx+1,j,k);

      v(cellsx+2,j,k) = v(2,j,k);
      v(cellsx+3,j,k) = v(3,j,k);

    }


    for(int i=0; i<cellsx+4; i++){
      v(i,0,k) = v(i,cellsy,k);
      v(i,1,k) = v(i,cellsy+1,k);

      v(i,cellsy+2,k) = v(i,2,k);
      v(i,cellsy+3,k) = v(i,3,k);
    }
   }
return v;
}

Tensor<double,3> Solver::transmissive_boundary(Tensor<double,3 >v){

      for(int k=0; k<4; k++){
    for(int j=0; j<cellsy+4; j++){
      v(0,j,k) = v(2,j,k);
      v(1,j,k) = v(2,j,k);

      v(cellsx+2,j,k) = v(cellsx+1,j,k);
      v(cellsx+3,j,k) = v(cellsx+1,j,k);

    }


    for(int i=0; i<cellsx+4; i++){
      v(i,0,k) = v(i,2,k);
      v(i,1,k) = v(i,2,k);

      v(i,cellsy+2,k) = v(i,cellsy+1,k);
      v(i,cellsy+3,k) = v(i,cellsy+1,k);
    }
   }
return v;
}

void Solver::ghost_boundary(){

  double normal_x,normal_y;
  VectorXf u_fluid;
  u_fluid.resize(4);
  //std::ofstream map("level_set.dat");
  VectorXf u_rigid;
  u_rigid.resize(4);


  for (int i =2 ; i<cellsx+3;i++){
    for(int j=2;j<cellsy+3;j++){

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

        }

}  


        double normal_moving = v_x_rot*normal_x + v_y_rot*normal_y;
       //double normal_moving = 1*normal_x + -1*normal_y;
       //double tangential_moving_x = v_x_rot - normal_moving*normal_x, tangential_moving_y = -normal_moving*normal_y;

       double normal_velocity_real = ((u_fluid(1)/u_fluid(0)))*normal_x + (u_fluid(3)/u_fluid(0))*normal_y;
       double tangential_velocity_x_real = ((u_fluid(1)/u_fluid(0))) - normal_velocity_real*normal_x , tangential_velocity_y_real = (u_fluid(3)/u_fluid(0)) - normal_velocity_real*normal_y;

       double normal_velocity_rigid = -(normal_velocity_real - 2*normal_moving);
       double tangential_velocity_x_rigid = tangential_velocity_x_real, tangential_velocity_y_rigid =tangential_velocity_y_real;
// HLLC  Right is the real liquid left is the rigid body

      double total_velocity_r = pow((u_fluid(1)/u_fluid(0)),2) + pow(u_fluid(3)/u_fluid(0),2);
      double pressure_r = ((u_fluid(2)-(0.5*u_fluid(0)*total_velocity_r))*(gamma-1));
      double cs_r = std::sqrt((gamma * (pressure_r)) /u_fluid(0));

      double total_velocity_l = pow((u_fluid(1)/u_fluid(0)),2) + pow(u_fluid(3)/u_fluid(0),2);
      double pressure_l = ((u_fluid(2)-(0.5*u_fluid(0)*total_velocity_l))*(gamma-1));
      double cs_l = std::sqrt((gamma * (pressure_l)) /u_fluid(0));



      double a_bar = 0.5*(cs_r+cs_l);
      double fi_bar = 0.5*(u_fluid(0)+u_fluid(0));
      double p_pvrs = 0.5*(pressure_l+pressure_r) - 0.5*(normal_velocity_rigid-normal_velocity_real)*a_bar*fi_bar;
      double pressure_star = std::max(0.,p_pvrs);
      double qr;
      double ql;

      if(pressure_star <= pressure_r){

        qr = 1.;

      }else if (pressure_star > pressure_r){
        qr = pow(1+(((gamma+1)/(2*gamma))*((pressure_star/(pressure_r))-1)),0.5);
      }


      if(pressure_star <= pressure_l){

        ql = 1.;

      }else if (pressure_star > pressure_l){
        ql = pow(1+(((gamma+1)/(2*gamma))*((pressure_star/(pressure_l))-1)),0.5);
      }

      //double s_r = std::max(abs(velocity_x_r) + cs_r,abs(velocity_x_l)+cs_l);
      //double s_l = -s_r;

      double s_l = normal_velocity_real - cs_l*ql;
      double s_r = normal_velocity_rigid + cs_r*qr;

      //double s_r = std::min(abs(normal_velocity_rigid) + cs_r,abs(normal_velocity_real)+cs_l);
      //double s_l = -s_r;

      double v_star = (pressure_r - pressure_l + (u_fluid(0)*normal_velocity_real*(s_l-normal_velocity_real)) - (u_fluid(0)*normal_velocity_rigid*(s_r-normal_velocity_rigid)))/(u_fluid(0)*(s_l-normal_velocity_real) - u_fluid(0)*(s_r-normal_velocity_rigid));
    
      double density_star = u_fluid(0)*((s_r - normal_velocity_rigid)/(s_r - v_star));
      double density_velocity_star = density_star*v_star;
      double energy_star = density_star * ((u_fluid(2))/u_fluid(0) + (v_star-normal_velocity_rigid)*(v_star + ((pressure_r)/(u_fluid(0)*(s_r-normal_velocity_rigid)))));
      double velocity_x_star = (density_velocity_star/density_star)*normal_x + tangential_velocity_x_real, velocity_y_star = (density_velocity_star/density_star)*normal_y + tangential_velocity_y_real;


      u_fluid(0) = density_star;
      u_fluid(1) = density_star*velocity_x_star;
      u_fluid(2) = energy_star;
      u_fluid(3) = density_star*velocity_y_star;

      //std::cout << u_fluid(3) << "\n";

      v(i,j,0) = u_fluid(0);
      v(i,j,1) = u_fluid(1);
      v(i,j,2) = u_fluid(2);
      v(i,j,3) = u_fluid(3);

      //std::cout << v(i,j,0) << " " <<v(i,j,1) << " " <<v(i,j,2) << " " <<v(i,j,3) << " " << loops << "\n";
       // set the values in the interfacial cells in the domain
      //level_map(i,j) = 2;
      //std::cout << normal_velocity_rigid << "\n";

      }
    }
  }

for (int i =2 ; i<cellsx+3;i++){
    for(int j=2;j<cellsy+3;j++){



      if (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0){ 
          //level_map(i,j) = pow(10,100);
           //v(i,j,0) = 1;
           //v(i,j,1) = 1;
           //v(i,j,2) = 1;
           //v(i,j,3) = 1;

      }
    }
  }


// sweeping logic

for(int times = 0 ; times < 1;times++){
    for (int i =2 ; i<cellsx+3;i++){
      for(int j=2;j<cellsy+3;j++){

       // This logic basically selects everything that is NOT an interface and that its level set function is bigger than 0 AND it requires that the level set function is increasing, it might not work with level set functions that have negative values inside the rigid body



         if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i-1,j) < level_set(i,j))){

           double  normal_x = ((level_set(i+1,j) - level_set(i-1,j))/(2*dx)), normal_y = ((level_set(i,j+1) - level_set(i,j-1))/(2*dy));

           double Qx,Qy;

         for(int k = 0; k<4; k++ ){

          if (level_set(i+1,j) < level_set(i-1,j)){
            Qx = v(i+1,j,k); 
          }else{
            Qx = v(i-1,j,k);
          }
          
          if (level_set(i,j+1) < level_set(i,j-1)){
            Qy = v(i,j+1,k); 
          }else{
            Qy = v(i,j-1,k);
          }
          v(i,j,k) = ((abs(normal_x)*Qx)/dx + ((abs(normal_y)*Qy)/dy))/((abs(normal_x))/dx + abs(normal_y)/dy);  

          //std::cout << v(i,j,0) << "\n";


         
         }
        }

      }
    }






        for (int i =2 ; i<cellsx+3;i++){
       for(int j=2;j<cellsy+3;j++){

          if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i,j) > level_set(i,j-1))){

           double  normal_x = ((level_set(i+1,j) - level_set(i-1,j))/(2*dx)), normal_y = ((level_set(i,j+1) - level_set(i,j-1))/(2*dy));

        double Qx,Qy;

         for(int k = 0; k<4; k++ ){

          if (level_set(i+1,j)< level_set(i-1,j-1)){
            Qx = v(i+1,j,k); 
          }else{
            Qx = v(i-1,j,k);
          }
          
          if (level_set(i,j+1)< level_set(i,j-1)){
            Qy = v(i,j+1,k); 
          }else{
            Qy = v(i,j-1,k);
          }
          
          v(i,j,k) = ((abs(normal_x)*Qx)/dx + ((abs(normal_y)*Qy)/dy))/((abs(normal_x))/dx + abs(normal_y)/dy);  


         
         }

         }
         }
         }


    for (int i =cellsx+2 ; i>2;i--){
      for(int j=cellsy+2;j>2;j--){

           if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i,j) > level_set(i,j+1))){
            
            
          double  normal_x = ((level_set(i+1,j) - level_set(i-1,j))/(2*dx)), normal_y = ((level_set(i,j+1) - level_set(i,j-1))/(2*dy));
         
         
         double Qx,Qy;

         for(int k = 0; k<4; k++ ){

          if (level_set(i+1,j)< level_set(i-1,j)){
            Qx = v(i+1,j,k); 
          }else{
            Qx = v(i-1,j,k);
          }
          
          if (level_set(i,j+1)< level_set(i,j-1)){
            Qy = v(i,j+1,k); 
          }else{
            Qy = v(i,j-1,k);
          }
          
          v(i,j,k) = ((abs(normal_x)*Qx)/dx + ((abs(normal_y)*Qy)/dy))/((abs(normal_x))/dx + abs(normal_y)/dy);  
       

         
         }
          }
          }
         
          }

 

     for (int i = cellsx+2 ; i>2;i--){
       for(int j=cellsy+2;j>2;j--){

          if( (!(((level_set(i,j)*level_set(i-1,j) < 0) || (level_set(i,j)*level_set(i,j-1) < 0)) && level_set(i,j) > 0 || ((level_set(i,j)*level_set(i+1,j) < 0) || (level_set(i,j)*level_set(i,j+1) < 0)) && level_set(i,j) > 0)  && level_set(i,j) > 0) && (level_set(i,j) > level_set(i+1,j))){

           double  normal_x = ((level_set(i+1,j) - level_set(i-1,j))/(2*dx)), normal_y = ((level_set(i,j+1) - level_set(i,j-1))/(2*dy));

         double Qx,Qy;

         for(int k = 0; k<4; k++ ){

          if (level_set(i+1,j)< level_set(i-1,j)){
            Qx = v(i+1,j,k); 
          }else{
            Qx = v(i-1,j,k);
          }
          
          if (level_set(i,j+1)< level_set(i,j-1)){
            Qy = v(i,j+1,k); 
          }else{
            Qy = v(i,j-1,k);
          }
          
          v(i,j,k) = ((abs(normal_x)*Qx)/dx + ((abs(normal_y)*Qy)/dy))/((abs(normal_x))/dx + abs(normal_y)/dy);  
      

         
         }

        
         }
         }
         }

}

  //  std::ofstream output("output.dat");

  // for(int i = 2; i < cellsx+2; i++ ){
  //   for(int j=2; j < cellsy+2; j++){

  //   double x = x0 + (i-1)*dx;
  //   double y = y0 + (j-1)*dy;

  //   double div_p = (v(i+1,j,0)-v(i-1,j,0))/(2*dx) + (v(i,j+1,0)-v(i,j-1,0))/(2*dy);
  //   double mock_shlieren = std::exp((-20*abs(div_p))/(1000*v(i,j,0)));

  //   output << x << " " << y << " " << v(i,j,0) << " "  << mock_shlieren << " " << div_p << "\n";
  //   }
  //   output << " " << "\n";
  // }

         


 }

MatrixXd Solver::evolve_level_set(MatrixXd set,double vx,double vy){


  for (int i=2; i<cellsx+2;i++){
        for (int j = 2 ; j<cellsy+2;j++){

          double x = x0 + (i-0.5)*dx;
          double y = y0 + (j-0.5)*dy;

          double gradient_x,gradient_y;

           if(vx <0){

             gradient_x = (set(i+1,j)-set(i,j));

           }else if (vx > 0){

             gradient_x = (set(i,j)-set(i-1,j));
           }

           if(vy <0){

             gradient_y = (set(i,j+1)-set(i,j));

           }else if (vy > 0){

             gradient_x = (set(i,j)-set(i,j-1));
           }

          level_set_new(i,j) = set(i,j)-dt*(vx*(gradient_x/dx) + vy*(gradient_y/dy));

          }



        }

          return level_set_new;
      }

double Solver::compute_time_step() {


// Compute the time step for each iteration

  MatrixXd a;
  a.resize(cellsx+4,cellsy+4);
  //MatrixXd abs_v = a;


  for (int i=0 ; i < cellsx+4; i++){
    for(int j=0; j<cellsy+4; j++){

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
 for (int i =0 ; i< cellsx+3;i++){
  for(int j=0; j<cellsy+3;j++){
  
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
for (int i =0 ; i< cellsx+3;i++){
  for(int j=0; j<cellsy+3;j++){

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
        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){

        delta_downwind(i,j,k) = v(i+1,j,k)-v(i,j,k);
          
          }
        }

      for(int i =1;i<cellsx+3;i++){
        for(int j=1; j<cellsy+3;j++){
        delta_upwind(i,j,k) = v(i,j,k)-v(i-1,j,k);
        }
      }

      for(int i =1;i<cellsx+3;i++){
        for(int j=1; j<cellsy+3;j++){

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
        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dx)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_x(v_L_half);
    v_R_flux = euler_flux_x(v_R_half);

    for (int i = 1 ; i < cellsx+2; i++){
      for (int j =1 ; j < cellsy+2 ; j++){


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

        qr = 1.;

      }else if (pressure_star > pressure_r){
        qr = pow(1+(((gamma+1)/(2*gamma))*((pressure_star/(pressure_r))-1)),0.5);
      }


      if(pressure_star <= pressure_l){

        ql = 1.;

      }else if (pressure_star > pressure_l){
        ql = pow(1+(((gamma+1)/(2*gamma))*((pressure_star/(pressure_l))-1)),0.5);
      }

      //double s_r = std::max(abs(velocity_x_r) + cs_r,abs(velocity_x_l)+cs_l);
      //double s_l = -s_r;

      double s_l = velocity_x_l - cs_l*ql;
      double s_r = velocity_x_r + cs_r*qr;

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

        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){
            delta_downwind(i,j,k) = v(i,j+1,k)-v(i,j,k);
          }
        }

        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){

            delta_upwind(i,j,k) = v(i,j,k)-v(i,j-1,k);
          }
        }

        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){

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
        for(int i =1;i<cellsx+3;i++){
          for(int j=1; j<cellsy+3;j++){

        v_L_half(i,j,k) = v_L(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        v_R_half(i,j,k) = v_R(i,j,k) - 0.5*(dt/dy)*(v_R_flux(i,j,k)-v_L_flux(i,j,k));

        }
      }
    }

    v_L_flux = euler_flux_y(v_L_half);
    v_R_flux = euler_flux_y(v_R_half);




    for (int i = 1 ; i < cellsx+2; i++){
      for (int j =1 ; j < cellsy+2 ; j++){

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

        qr = 1.;

      }else if(pressure_star > pressure_r) {
        qr = pow(1+(((gamma+1)/(2*gamma))*((pressure_star/(pressure_r))-1)),0.5);
      }


      if(pressure_star <= pressure_l){

        ql = 1.;

      }else if (pressure_star > pressure_l){

        ql = pow(1+((gamma+1)/(2*gamma))*(((pressure_star/(pressure_l)))-1),0.5);

      }

      //double s_r = std::min(abs(velocity_y_r) + cs_r,abs(velocity_y_l)+cs_l);
      //double s_l = -s_r;


      double s_l = velocity_y_l - cs_l*ql;
      double s_r = velocity_y_r + cs_r*qr;


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



void Solver::velocity_field(double time){

  double theta = ang_vel*time;
  double row,col;
  double max = level_set.maxCoeff(&row,&col);

  double max_x = x0 + (row-0.5)*dx;
  double max_y = y0 + (col-0.5)*dy;

  double angle = atan2(0.5-max_y,0.5-max_x);



  v_x_rot = -std::sin(angle);
  v_y_rot = std::cos(angle);
}

MatrixXd Solver::advect_level_set(MatrixXd set,double vx,double vy,double time){


  // double row,col;
  // double max = level_set.maxCoeff(&row,&col);
  // double velocity = 0.2;

  // double max_x = x0 + (row-0.5)*dx;
  // double max_y = y0 + (col-0.5)*dy;

  // double r_x = 0.5 - max_x, r_y = 0.5 - max_y;
  // //std::cout << max_x << " " << max_y << std::endl;
  // double mag_r= std::sqrt(pow(r_x,2) + pow(r_y,2));

  // double norm_r_x = r_x/mag_r, norm_r_y = r_y/mag_r; 

  // //std::cout << norm_r_x << " " << norm_r_y << std::endl;

  // double acc_x = norm_r_x*pow(velocity,2)/0.3 , acc_y = norm_r_y*pow(velocity,2)/0.3;

  //  //std::cout << acc_x << " " << acc_y << std::endl;

  // v_x_rot = v_x_rot+acc_x,v_y_rot = v_y_rot+acc_y ;

  // std::cout << v_x_rot << " " << v_y_rot << std::endl;

  // double x_centre = 0.2+v_x_rot, y_centre = 0.5 + v_y_rot;

  // //std::cout << x_centre << " " << y_centre << std::endl; 
  
 
  double adv_x = -0.3*std::cos(ang_vel*time);
  double adv_y = -0.3*std::sin(ang_vel*time);


  v_x_rot =  0.3*ang_vel*std::sin(ang_vel*time);
  v_y_rot = -0.3*ang_vel*std::cos(ang_vel*time);


 for(int i=0; i < cellsx+4; i++){ // changed from v.cols to cells+2
      for (int j =0 ; j<cellsy+4; j++){

      double x = x0 + (i-0.5)*dx;
      double y = y0 + (j-0.5)*dy;

  level_set_new(i,j) = (0.15-std::sqrt(pow((x-0.5-adv_x),2)+pow((y-0.5-adv_y),2)));


      }}

return level_set_new;

}

void Solver::simulate(){
 
 //v_x_rot = 0.;
  //v_y_rot = 0.;


  //initialise_input_explosion();eal = ((u_fluid(1)/u_fluid(0))) - n
  //initialise_Sod();
  initialise_input();
  //moving_rigid_body();
  //moving_rigid_body_moving();
  //rotating_rigid_body();
  //ghost_boundary(level_set);
  //ghost_boundary(level_set);
  std::ofstream output_animation("output_animation.dat");
  std::ofstream map("level_set.dat");
  
  
  double t = 0;


  do{
   //double angle = ang_vel*t;
  v_x_rot = -0.;
  v_y_rot = 0;
   ghost_boundary();
   v = transmissive_boundary(v);
   dt = compute_time_step();
   //level_set = evolve_level_set(level_set,v_x_rot,v_y_rot);
   //level_set = advect_level_set(level_set,v_x_rot,v_y_rot,t);
   std::cout << dt << t << std::endl;

   t = t+dt;




     flux = flux_all(v,'x');
     flux_initial = euler_flux_x(v);
   
     for (int i=2; i<cellsx+2;i++){
      for (int j = 2 ; j<cellsy+2;j++){
        for (int k =0 ; k<4;k++){
          v_new(i,j,k) = v(i,j,k) - (dt/dx) * (flux(i,j,k)-flux(i-1,j,k));
        }
      }
     }

      v_new = transmissive_boundary(v_new);

      flux = flux_all(v_new,'y');
      flux_initial = euler_flux_y(v_new);

      for (int i=2; i<cellsx+2;i++){
        for (int j = 2 ; j<cellsy+2;j++){

    double velocity_x = (v(i,j,1)/v(i,j,0));
    double velocity_y = v(i,j,3)/v(i,j,0);
    double kinetic_energy = 0.5*v(i,j,0)*(pow(velocity_x,2)+pow(velocity_y,2));
    double pressure = (v(i,j,2)-kinetic_energy)*(gamma-1); 

              double x = x0 + (i-0.5)*dx;
              double y = y0 + (j-0.5)*dy;


          for (int k =0 ; k<4;k++){
              v_new(i,j,k) = v_new(i,j,k) - (dt/dy) * (flux(i,j,k)-flux(i,j-1,k));       
            }
           double div_p = (v(i+1,j,0)-v(i-1,j,0))/(2*dx) + (v(i,j+1,0)-v(i,j-1,0))/(2*dy);
            double mock_shlieren = std::exp((-20*abs(div_p))/(1000*v(i,j,0)));

              
  


          output_animation << t << " " << x << " " << y << " " << v_new(i,j,0) << " " << mock_shlieren << " "  << pressure << " " <<  level_set(i,j)<< "\n";

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

  for(int i = 2; i < cellsx+2; i++ ){
    for(int j=2; j < cellsy+2; j++){

    double x = x0 + (i-1)*dx;
    double y = y0 + (j-1)*dy;

    double div_p = (v(i+1,j,0)-v(i-1,j,0))/(2*dx) + (v(i,j+1,0)-v(i,j-1,0))/(2*dy);
    double mock_shlieren = std::exp((-20*abs(div_p))/(1000*v(i,j,0)));

    output << x << " " << y << " " << v(i,j,0) << " "  << mock_shlieren << " " << div_p << "\n";
    }
    output << " " << "\n";
  }
}

int main(){
 Solver solver;
 solver.simulate(); 
 solver.save_plot();
}
