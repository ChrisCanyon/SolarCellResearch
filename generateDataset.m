function [inputs, labels] = generateDataset(cellsPerSensor,s, threadPool)
    inputs = zeros(s,2);
    labels = [];

    parfor (i = 1:s, threadPool)
        avgTemp = 0;
        avgIrrad = 0;
        temp = 50*rand; %look into making temp range evenly across the board
        P_total = zeros(1,38001);

        cell = pv_obj;
        for j=1:cellsPerSensor
            cell.temp = temp - (5*rand);
            cell.irrad = 100*rand;
            cell.set_vals();
            P_total = P_total + cell.P;
            avgTemp = avgTemp + cell.temp;
            avgIrrad = avgIrrad + cell.irrad;
        end
        %calculate average temp and irrad
        avgTemp = avgTemp/cellsPerSensor;
        avgIrrad = avgIrrad/cellsPerSensor;

        %find real MPPT for sensor
        [~,I] = max(P_total);
        V_maxP = cell.V(I);

        %insert data into array
        inputs(i,:) = [avgTemp, avgIrrad];
        labels(i) = V_maxP;
    end    
end
