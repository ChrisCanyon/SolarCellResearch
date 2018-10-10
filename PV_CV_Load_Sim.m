function [P,I,V,Req,load_curve]= PV_CV_Load_Sim(irrad,temp,Rload,D)


%This Program simulates the solar cell/loaded ideal converter in constant
%conduction mode. The inputs are the operating irradiance, temperature,
%load connected to the converter, and the duty cycle of the sepic, cuk, or 
%zeta converter.

PV= pv_obj;                               %PV cell object
PV.irrad= irrad;                          %PV irradiance
PV.temp= temp;                            %PV temperature
PV.set_vals;                              %Generate PV based on values
load_curve= PV.V./PV.I;                   %Generate matched load curve

Load_Transfer_Factor= (D/(1-D))^2;        %Generate load transfer factor
Rloadin= Rload*Load_Transfer_Factor;      %Load apparent at PV Cell       
 
[ta,index]= min(abs(load_curve-Rloadin));  %Find closest value on matched load curve
Req= load_curve(index);                   %Find index of apparent load 


I= PV.I(index);                           %Current at the loaded PV terminals
V= PV.V(index);                           %Voltage at the loaded PV terminals
P= I*V;                                   %Power Generated at loaded PV terminals