%{
sunny_array_size = 1;
cloudy_array_size = 1;

output = [];

parfor (x = 0:100,8)
    temp = x;
    irrad = 100;
    irrad_sunny = irrad;
    %create solar array
    solar_array_sunny(sunny_array_size) = pv_obj;
    solar_array_cloudy(cloudy_array_size) = pv_obj;

    bad = true;
    while bad
        if irrad < 0
            break;
        end
        %initialize solar cells
        solar_array_sunny = initialize_pv_array(solar_array_sunny, irrad_sunny, temp);
        solar_array_cloudy = initialize_pv_array(solar_array_cloudy, irrad, temp);
        %solar_array_cloudy = initialize_pv_array(solar_array_cloudy, irrad, temp-(5 - (5*irrad/100)));

        %assuming the size of .I is the same between solar cells  
        I_size = size(solar_array_sunny(1).I);
        P_total = zeros(1,I_size(2));
        V = solar_array_sunny(1).V;

        %sum each solar cell's currents
        for i = 1:sunny_array_size
            P_total = P_total + solar_array_sunny(i).P;
        end
        for i = 1:cloudy_array_size
            P_total = P_total + solar_array_cloudy(i).P;
        end

        [peaks, locations] = findpeaks(P_total);

        num_peaks = size(peaks);
        if num_peaks(2) > 1
           %we found params to create peaks
           bad = false;
        else
            irrad = irrad - 1;
        end
    end
    %plot(V, P_total);
    X = sprintf('temp: %d irrad cloudy: %d',temp, irrad);
    disp(X);
end

for i = 1:sunny_array_size
    plot(V, solar_array_sunny(i).P);
    figure
end
for i = 1:cloudy_array_size
    plot(V, solar_array_cloudy(i).P);
    figure
end
%}




