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
V.del= 0;
P.prev= 0;
P.curr= P_initial;
P.del= 0;
epochs= 0;

Power.ideal= [];
Power.mat= [];
Power.ch_mat= [];
Error.V= [];
Error.P=[];
Real_Time.P= [];
Real_Time.V= [];
Real_Time.D= [];

while(epochs<min_epochs)
    %Voltage for current loop
    V.curr= V.curr+ V.del;
    %Get the current Power and save the previous Power
    [~,trend_PV]= PV_CV_Control(irrad,temp,V.curr,D,Rload);
    D= trend_PV.D(end);
    P.prev= P.curr;
    P.curr= trend_PV.P(end);
    P.del= P.curr-P.prev;
    Power.ch_mat= [Power.ch_mat P.del];
    
    Derivative_P_Over_V= P.del/V.del;
    
    
    %Fuzzy Control of change in voltage
    V.del= fuzzy_MPP_Controller(Derivative_P_Over_V);
    
    
    epochs= epochs+1
    
    p= pv_obj;
    p.irrad= irrad;
    p.temp= temp;
    p.set_vals;
    
    Power.ideal= [Power.ideal p.P_mppt*ones(1,length(trend_PV.P))];
    Power.mat= [Power.mat trend_PV.P];
    Error.V= [Error.V V.curr-p.V_mppt];
    Error.P= [Error.P 100*(trend_PV.P-p.P_mppt)/p.P_mppt];
    
    temp= temp- (rand-0.3)*50/min_epochs;
    irrad= irrad- (rand-0.3)*100/min_epochs;
    
    Real_Time.V= [Real_Time.V trend_PV.V_pv];
    Real_Time.P= [Real_Time.P trend_PV.P];
    Real_Time.D= [Real_Time.D trend_PV.D];
    
end


plot(abs(Error.P))
title('Fuzzy MPPT - Error in Max Power')
xlabel('epochs')
ylabel('Percent Error')
ylim([0 8])
figure 
plot(Power.mat, '-')
xlabel('epochs')
ylabel('Watts')
hold on
title('Fuzzy MPPT - Ideal and Actual Power Out of PV')
plot(Power.ideal, '.')
legend('Actual Power Output','Ideal Maximum Power')

figure 
plot(Real_Time.D)
xlabel('epochs')
ylabel('Duty Ratio')
title('Fuzzy MPPT - Duty Cycle')




