classdef pv_obj < handle
    
    %Instantiates a solar cell simulation object
    %Methods of initialization
    %1. Random - provide a size upon instantiation t=pv_obj(1);
    %2. Set    - Set the irradiance and temperature and call set_vals
    %2-cont'd  - t=pv_obj; t.irrad=100; t.temp=50; t.set_vals;
    
    properties
        P          %Array of the PV Power
        V          %Array of the PV Voltage
        I          %Array of the PV Current
        irrad      %Irradiance from 0 to 100% of 1000 insolation/meter
        temp       %Temperature in celcius
        V_mppt     %PV Voltage corresponding to the maximum power point
        P_mppt     %PV Power corresponding to the maximum power point
        V_oc       %PV Voltage corresponding to 0 amps
        I_sc       %PV Current corresponding to 0 volts
    end
    
    methods
        function obj= pv_obj(size)
            %Object Array Instatiate Function
            if(nargin~=0)
                
                obj(size)= pv_obj;
                for size= 1:size
                    obj(size).init();
                end
            end
        end
        
        function obj= init(obj)
            %Generate Random solar cell
            obj.irrad= (100-.54)*rand+.54;  %0.54 ensures that irradiance is not too low :)
            obj.temp= 100*rand;
            obj.set_vals;
        end
        
        function obj= mppt(obj)
            [obj.P_mppt,index]= max(obj.P);
            obj.V_mppt= obj.V(index);
        end
        
        function plot_mpp(obj)
            %Plot Power Curve and MPPT
            hold on
            
            term= find(obj.P<0.01,3);
            term_index= find(term>100,1);
            stop= term(term_index);
            
            ylim([0 max(obj.P)+5])
            plot(obj.V(1:stop),obj.P(1:stop))
            plot(obj.V_mppt,obj.P_mppt,'o')
            hold off
        end
        
        function plot_oc_sc(obj)
            %Plot Open circuit Voltage and Short Circuit Current
            hold on
            plot(obj.V,obj.I)
            plot(obj.V_oc,0,'o')
            plot(0,obj.I_sc,'o')
            hold off
        end
        
        function obj= set_vals(obj)
            %Generate Solar Cell Based on the set irradiance and temp
            [obj.P,obj.I,obj.V]= PV_Sim(obj.irrad,obj.temp);
            obj.mppt;
            obj.V_oc= obj.V(find(obj.I==min(obj.I),1));
            [ta,index]= max(obj.I);
            obj.I_sc= obj.I(index);
            
            if isempty(obj.P) || isempty(obj.V) || isempty(obj.I)
                error('irradiance is too low')
            end
        end
    end
    
end

