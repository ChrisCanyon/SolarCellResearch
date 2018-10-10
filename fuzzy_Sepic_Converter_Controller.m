function del_Duty_Cycle= fuzzy_Sepic_Converter_Controller(voltage_err)

%Generate Fuzzy Logic Controller or Load from memory

if exist('CV_Fuzzy.mat', 'file') ~= 2
    %Fuzzy Controller Setup
    CV_Fuzzy = newfis('CV_Fuzzy','mamdani','min','max','min','max','centroid');
    CV_Fuzzy = addvar(CV_Fuzzy,'input','Error in Voltage',[-50 50]);
    CV_Fuzzy = addmf(CV_Fuzzy,'input',1,'NL','trimf',[-50,-10,-5]);
    CV_Fuzzy = addmf(CV_Fuzzy,'input',1,'NS','trimf',[-10,-5,0]);
    CV_Fuzzy = addmf(CV_Fuzzy,'input',1,'Z','trimf',[-5,0,5]);
    CV_Fuzzy = addmf(CV_Fuzzy,'input',1,'PS','trimf',[0,5,10]);
    CV_Fuzzy = addmf(CV_Fuzzy,'input',1,'PL','trimf',[5,10,50]);
    CV_Fuzzy = addvar(CV_Fuzzy,'output','Delta Duty Cycle',[-0.1 .1]);
    CV_Fuzzy = addmf(CV_Fuzzy,'output',1,'NL','trimf', [-0.27,-0.2,-0.05]);
    CV_Fuzzy = addmf(CV_Fuzzy,'output',1,'NS','trimf', [-0.1,-0.05,-0.02]);
    CV_Fuzzy = addmf(CV_Fuzzy,'output',1,'Z','trimf', [-0.07,0,0.07]);
    CV_Fuzzy = addmf(CV_Fuzzy,'output',1,'PS','trimf', [0.02,0.05,0.1]);
    CV_Fuzzy = addmf(CV_Fuzzy,'output',1,'PL','trimf', [0.05,0.2,0.27]);
    CV_Fuzzy = addrule(CV_Fuzzy,[1 1 1 1; 2 2 1 1; 3 3 1 1; 4 4 1 1; 5 5 1 1]);
    save('CV_Fuzzy.mat', 'CV_Fuzzy')
else
    load('CV_Fuzzy.mat', 'CV_Fuzzy')
end

%Condition the crisp input
if voltage_err> 50
    voltage_err= 50;
elseif voltage_err< -50
    voltage_err= -50;
end

%Output
del_Duty_Cycle= evalfis(voltage_err,CV_Fuzzy);






