function del_Voltage= fuzzy_MPP_Controller(Derivative_P_Over_V)

%Generate Fuzzy Logic Controller or Load from memory

if exist('MPP_Fuzzy.mat', 'file') ~= 2
    %Fuzzy Controller Setup
    MPP_Fuzzy = newfis('MPP_Fuzzy','mamdani','min','max','min','max','centroid');
    MPP_Fuzzy = addvar(MPP_Fuzzy,'input','Delta Power/Delta Voltage',[-20 20]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'input',1,'NL','trimf',[-36,-20,-10]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'input',1,'NS','trimf',[-20,-10,0]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'input',1,'Z','trimf',[-10,0,10]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'input',1,'PS','trimf',[0,10,20]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'input',1,'PL','trimf',[10,20,36]);
    MPP_Fuzzy = addvar(MPP_Fuzzy,'output','Delta Voltage',[-2 2]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'output',1,'NL','trimf',[-3.60000000000000,-2.4,-1]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'output',1,'NS','trimf',[-2.4,-1,0]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'output',1,'Z','trimf',[-.5,0,.5]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'output',1,'PS','trimf',[0,1,2.4]);
    MPP_Fuzzy = addmf(MPP_Fuzzy,'output',1,'PL','trimf',[1,2.4,3.60000000000000]);
    MPP_Fuzzy = addrule(MPP_Fuzzy,[1 1 1 1; 2 2 1 1; 3 3 1 1; 4 4 1 1; 5 5 1 1]);
    save('MPP_Fuzzy.mat', 'MPP_Fuzzy')
else
    load('MPP_Fuzzy.mat', 'MPP_Fuzzy')
end

%Condition the crisp input
if isnan(Derivative_P_Over_V)
    Derivative_P_Over_V= 0;
elseif Derivative_P_Over_V>20
    Derivative_P_Over_V= 20;
elseif Derivative_P_Over_V<-20
    Derivative_P_Over_V= -20;
end

%Output
del_Voltage= evalfis(Derivative_P_Over_V,MPP_Fuzzy);






