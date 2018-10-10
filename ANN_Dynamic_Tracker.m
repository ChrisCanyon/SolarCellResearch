clc
clear all
close all

irrad= 100;
temp= 30;
Rload= 20;
D= 0.5;

[P_initial,~,V_initial,~,~]= PV_CV_Load_Sim(irrad,temp,Rload,D);

min_epochs= 1e3;
V.curr= V_initial;
P.curr= P_initial;
epochs= 0;

trend_Power.ideal= [];
trend_Power.mat= [];
Err.V= [];
Err.P=[];
Real_Time.P= [];
Real_Time.V= [];
Real_Time.D= [];

while(epochs<min_epochs)
    %Voltage for current loop
    %Get the current Power and save the previous Power
    [~,trend_PV]= PV_CV_Control(irrad,temp,V.curr,D,Rload);
    D= trend_PV.D(end);
    P.curr= trend_PV.P(end);
    
    ANN_Estimation= ANN_V_MPP_Estimator(irrad,temp);
    V.curr= ANN_Estimation;
  
    
    
    
    epochs= epochs+1
    
    p= pv_obj;
    p.irrad= irrad;
    p.temp= temp;
    p.set_vals;
    
    trend_Power.ideal= [trend_Power.ideal p.P_mppt*ones(1,length(trend_PV.P))];
    trend_Power.mat= [trend_Power.mat trend_PV.P];
    Err.V= [Err.V V.curr-p.V_mppt];
    Err.P= [Err.P 100*(trend_PV.P-p.P_mppt)/p.P_mppt];
    
    temp= temp- (rand-0.3)*50/min_epochs;
    irrad= irrad- (rand-0.3)*100/min_epochs;
    
    Real_Time.V= [Real_Time.V trend_PV.V_pv];
    Real_Time.P= [Real_Time.P trend_PV.P];
    Real_Time.D= [Real_Time.D trend_PV.D];
    
end


plot(abs(Err.P))
title('ANN MPPT - Error in Max Power')
xlabel('epochs')
ylabel('Percent Error')
ylim([0 .01])
figure 
plot(trend_Power.mat, '-')
xlabel('epochs')
ylabel('Watts')
hold on
title('ANN MPPT - Ideal and Actual Power Out of PV')
plot(trend_Power.ideal, '.')
legend('Actual Power Output','Ideal Maximum Power')

figure 
plot(Real_Time.D)
xlabel('epochs')
ylabel('Duty Ratio')
title('ANN MPPT - Duty Cycle')




