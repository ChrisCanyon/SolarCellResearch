sunny_array_size = 1;
cloudy_array_size = 100;

temp = 100;
irrad = 100;

%create solar array
solar_array_sunny(sunny_array_size) = pv_obj;
solar_array_cloudy(cloudy_array_size) = pv_obj;

%initialize solar cells
solar_array_sunny = initialize_pv_array(solar_array_sunny, irrad, temp);
solar_array_cloudy = initialize_pv_array(solar_array_cloudy, irrad*.05, temp);


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

%{
for i = 1:sunny_array_size
    plot(V, solar_array_sunny(i).P);
    figure
end
for i = 1:cloudy_array_size
    plot(V, solar_array_cloudy(i).P);
    figure
end
%}

[peaks, locations] = findpeaks(P_total);
num_peaks = size(peaks);
plot(V, P_total);
disp(num_peaks);
disp(peaks);
disp(locations);

