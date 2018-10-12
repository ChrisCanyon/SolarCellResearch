array_size = 300;
solar_array = pv_obj(array_size);

%assuming the size of .I is the same between solar cells  
I_size = size(solar_array(1).I);
P_total = zeros(1,I_size(2));
V = solar_array(1).V;

%sum each solar cell's currents
for i = 1:array_size
    P_total = P_total + solar_array(i).P;
end 

plot(V, P_total);