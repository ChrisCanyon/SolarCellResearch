function del_Voltage= fuzzy_MPP_Controller_w_ANN(Derivative_P_Over_V,ANN_Fuzzy_Input)

%Generate Fuzzy Logic Controller or Load from memory

if exist('MPP_Fuzzy_w_ANN.mat', 'file') ~= 2 
    %Fuzzy Controller Setup
    MPP_Fuzzy_w_ANN = newfis('MPP_Fuzzy_w_ANN','mamdani','min','max','min','max','centroid');
    MPP_Fuzzy_w_ANN = addvar(MPP_Fuzzy_w_ANN,'input','Delta Power/Delta Voltage',[-20 20]);
    MPP_Fuzzy_w_ANN = addvar(MPP_Fuzzy_w_ANN,'input','ANN Estimated Power minus Actual Power (Watts)',[-10 10]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',1,'NL','trimf',[-36,-20,-10]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',1,'NS','trimf',[-20,-10,0]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',1,'Z','trimf',[-10,0,10]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',1,'PS','trimf',[0,10,20]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',1,'PL','trimf',[10,20,36]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',2,'MP-N','trapmf',  [-5,-5,-1,0]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'input',2,'MP-P','trapmf',  [0,1,5,5]);
    MPP_Fuzzy_w_ANN = addvar(MPP_Fuzzy_w_ANN,'output','Delta Voltage',[-2 2]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'output',1,'NL','trimf',[-3.60000000000000,-2.4,-1]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'output',1,'NS','trimf',[-2.4,-1,0]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'output',1,'Z','trimf',[-.5,0,.5]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'output',1,'PS','trimf',[0,1,2.4]);
    MPP_Fuzzy_w_ANN = addmf(MPP_Fuzzy_w_ANN,'output',1,'PL','trimf',[1,2.4,3.60000000000000]);
    MPP_Fuzzy_w_ANN = addrule(MPP_Fuzzy_w_ANN,[1 0 1 1 1; 2 0 2 1 1;...
        3 0 3 1 1; 4 0 4 1 1; 5 0 5 1 1; 3 1 2 1 1; 3 2 4 1 1]);
    save('MPP_Fuzzy_w_ANN.mat', 'MPP_Fuzzy_w_ANN')
else
    load('MPP_Fuzzy_w_ANN.mat', 'MPP_Fuzzy_w_ANN')
end

%Condition the crisp input
if isnan(Derivative_P_Over_V)
    Derivative_P_Over_V= -10;
elseif Derivative_P_Over_V>20
    Derivative_P_Over_V= 20;
elseif Derivative_P_Over_V<-20
    Derivative_P_Over_V= -20;
end

if ANN_Fuzzy_Input>10
    ANN_Fuzzy_Input= 10;
elseif ANN_Fuzzy_Input<-10
    ANN_Fuzzy_Input= -10;
end

%Output
del_Voltage= evalfis([Derivative_P_Over_V,ANN_Fuzzy_Input],MPP_Fuzzy_w_ANN);






