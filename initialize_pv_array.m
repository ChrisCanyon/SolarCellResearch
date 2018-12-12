function obj=initialize_pv_array(pv_array, irrad, temp)
    s = size(pv_array);
    for i= 1:s(2)
        cell = pv_array(i);
        cell.irrad = irrad;
        cell.temp = temp;
        pv_array(i) = cell.set_vals();    
    end
    obj = pv_array;
end

