array_size = 10;
solar_array = pv_obj(array_size);

%assuming the size of .I is the same between solar cells  
I_size = size(solar_array(1).I);

I_total = zeros(1,I_size);

%sum each solar cell's currents
for i= 1:array_size
   for j = 1:I_size
     I_total(j) = I_total(j) + solar_array(i).I(j);
   end
end

P = I_total.dot(solar_array(1).V);

plot(P+solar_array(1).V)