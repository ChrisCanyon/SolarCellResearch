function [inputs, labels] = generateDataset(cellsPerSensor,size)
    inputs = zeros(size,2);
    labels = [];
    for i = 1:size
        avgTemp = 0;
        avgIrrad = 0;
        temp = 50*rand; %look into making temp range evenly across the board
        P_total = zeros(1,38001);
 
        cells(cellsPerSensor) = pv_obj;
        for j = 1:cellsPerSensor
            cells(j).temp = temp - (5*rand);
            cells(j).irrad = 100*rand;
            cells(j).set_vals();
            P_total = P_total + cells(j).P;
            avgTemp = avgTemp + cells(j).temp;
            avgIrrad = avgIrrad + cells(j).irrad;
        end
        
        %calculate average temp and irrad
        avgTemp = avgTemp/cellsPerSensor;
        avgIrrad = avgIrrad/cellsPerSensor;
        
        %find real MPPT for sensor
        [~,I] = max(P_total);
        V_maxP = cells(1).V(I);
        
        %insert data into array
        
        inputs(i) = avgTemp;
        inputs(i,2) = avgIrrad;
        labels(i) = V_maxP;
    end    
end
