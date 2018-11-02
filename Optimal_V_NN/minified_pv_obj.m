classdef minified_pv_obj < handle
    properties
        irrad
        temp
        label
    end
    
    methods
        function obj =minified_pv_obj(s)
            disp(s);
            data = pv_obj(s);

            obj(s) = minified_pv_obj;
            for i = 1:s
                obj(i).irrad = data(i).irrad;
                obj(i).temp = data(i).temp;
             
                [peaks, locations] = findpeaks(data(i).P);
                peak_V = data(i).V(locations(2));
             
                obj(i).label = peak_V;
                disp(peak_V);
                plot(data(i).V, data(i).P);
           end
        end
    end
end

