function [epochs,trend_PV]= PV_CV_Control(irrad,temp,voltage,D_prev,Rload,max_epochs)

%Tracks the loaded PV voltage to the input voltage
%If convergence can't be reached, a fault will be executed
if nargin<6
    max_epochs= 1e3;
end
if nargin< 5
    error('not enough input arguments')
end

epochs= 0;

D= D_prev;                  %Previous Duty Cycle for continuous tracking
trend_PV.P= [];             %Power trend as voltage approaches desired
trend_PV.V_pv= [];          %Voltage trend as voltage approaches desired
trend_PV.I_pv= [];          %Current trend as voltage approaches desired
trend_PV.Req= [];           %Req trend as voltage approaches desired
trend_PV.D= [];             %Duty Cycle trend as voltage approaches desired 

[P,I_pv,V_pv,Req,~]= PV_CV_Load_Sim(irrad,temp,Rload,D); %Initial conditions
err= voltage-V_pv;                                       
trend_PV.V_pv=[trend_PV.V_pv V_pv];
trend_PV.I_pv=[trend_PV.I_pv I_pv];
trend_PV.Req=[trend_PV.Req Req];
trend_PV.P=[trend_PV.P P];
trend_PV.D=[trend_PV.D D];

while abs(err)> .001 && epochs<max_epochs
    D= D+ fuzzy_Sepic_Converter_Controller(err);
    [P,I_pv,V_pv,Req,~]= PV_CV_Load_Sim(irrad,temp,Rload,D);
    err= voltage-V_pv;
    trend_PV.V_pv=[trend_PV.V_pv V_pv];
    trend_PV.I_pv=[trend_PV.I_pv I_pv];
    trend_PV.Req=[trend_PV.Req Req];
    trend_PV.P=[trend_PV.P P];
    trend_PV.D=[trend_PV.D D];
    epochs= epochs+1;
end

if abs(err)>.05
    error('Failed to converge: Intednded voltage was %d, V_pv converged to %d increase the max_epochs', voltage,V_pv)
end
    