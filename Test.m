array_size = 1;
solar_array = pv_obj(array_size);

%assuming the size of .I is the same between solar cells  
I_size = size(solar_array(1).I);

I_total = zeros(1,I_size(2));
%sum each solar cell's currents
for i= 1:array_size
   for j = 1:I_size(2)
     I_total(j) = I_total(j) + solar_array(i).I(j);
   end
end

P = I_total.*solar_array(1).V;

plot(solar_array(1).V+P);