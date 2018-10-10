function Sim_Value= ANN_P_MPP_Estimator(irrad,temp)


if exist('ANN_P_mppt.mat', 'file') ~= 2
    %Generate ANN if ANN does not already exist
    
    cells= 1000;
    tests= 30;
    
    layers= 2;
    neurons= 7;
    
    %t= pv_obj(cells);
    t= pv_obj;
    t_up= pv_obj;
    t_rand= pv_obj;
    
    inputs= zeros(2,cells*3);
    targets= zeros(1,cells*3);
    for i= 1:cells
        
        t.irrad= (cells/i)-.4;
        t.temp= 100*i/cells;
        t.set_vals;
        t_up.irrad= 100*i/cells;
        t_up.temp= 100*i/cells;
        t_up.set_vals;
        t_rand.init;
        
        
        %inputs(1,i)= t(i).V_oc;
        %inputs(2,i)= t(i).I_sc;
        inputs(1,i*3)= t.temp;
        inputs(2,i*3)= t.irrad;
        inputs(1,i*3-1)= t_up.temp;
        inputs(2,i*3-1)= t_up.irrad;
        inputs(1,i*3-2)= t_rand.temp;
        inputs(2,i*3-2)= t_rand.irrad;
        targets(1,i*3)= t.P_mppt;
        targets(1,i*3-1)= t_up.P_mppt;
        targets(1,i*3-2)= t_rand.P_mppt;
    end
    
    
    structure= ones(1,layers);
    structure= structure.*neurons;
    
    
    % Training
    ANN_P_mppt = newff(inputs,targets,structure);
    ANN_P_mppt.trainParam.epochs = 1000;
    ANN_P_mppt.trainParam.goal = 1e-9;
    ANN_P_mppt.trainParam.min_grad = 1e-9;
    ANN_P_mppt.trainParam.max_fail = 500;
    ANN_P_mppt.performFcn = 'mse';
    ANN_P_mppt = train(ANN_P_mppt,inputs,targets);
    save('ANN_P_mppt.mat','ANN_P_mppt');
    
    %Generalize
    errors= zeros(1,tests);
    actual= zeros(1,tests);
    netout= zeros(1,tests);
    test= pv_obj;
    
    for i= 1:tests
        
        %     test.irrad= (tests/i)-1;
        %     test.temp= 100*i/tests;
        %     test.set_vals;
        test= pv_obj(1);
        
        actual(i)= test.P_mppt;
        %netout(i)= net([test.V_oc;test.I_sc;test.temp;test.irrad]);
        netout(i)= sim(ANN_P_mppt,[test.temp;test.irrad]);
        errors(i)= actual(i)-netout(i);
    end
    
    RMSE= ((sum(errors)^2)/length(errors))^0.5
    [actual;netout]
    
    % Results
    
    outputs= sim(ANN_P_mppt,inputs);
    figure; plot(targets,'r+'); hold on; plot(outputs,'o');
    title('ANN Training Set Validation'); legend('Target','Output');
    ylabel('Watts')
    xlabel('Training Point Number')
    
    figure; plot(errors,'o'); grid;
    title('Error in Generalized ANN');
    ylabel('Error in Watts')
    xlabel('Generalization Test Number')
else
    
    warning('off', 'all')
    load('ANN_P_mppt.mat', 'ANN_P_mppt');
end

Sim_Value= sim(ANN_P_mppt,[temp; irrad]);
