function [P_pvt,I_pvt,V_pvt] = PV_Sim(irradiance,temperature)

%Generates a PV Cell Power, Current, and Voltage curves
V_pvt= 0:0.001:38;
I_pvt= zeros(1,length(V_pvt));

Ns= 54;                          %Number of cells in array
k= 1.3806503*10^(-23);           %
q= 1.60217646*10^(-19);          %

TK= 273.15;                      %
I_pvn= 8.214;                    %Photovoltaic nominal current
KI= 0.0032;                      %
a = 1.3;                         %Diode quality factor
I_scn= 8.21;                     %Short circuit current
I_sol= I_scn;                    %Starting current for iterative solution
Vocn= 32.9;                      %Open circuit Voltage
Kv= -0.123;                      %
Rp= 415.405;                     %Shunt resistance
Rs= 0.221;                       %Series resistance
Tn= TK + 25;                     %Nominal temperature

T= TK + temperature;             %Conversion of celcius to kelvin
Sun= 10 * irradiance;            %Conversion of percentage to insolation

IRR= Sun/1000;                   %Conversion of insolation to irradiance
vt= Ns.*k.*T/q;                  %
Dt= (T-Tn);                      %
Ipv=(I_pvn+KI.*Dt).*(IRR);       %Photovoltaic Current
Io= (I_scn+KI.*Dt)/(exp((Vocn+Kv.*Dt)/(a.*vt))-1); %Diode Current

%Ion= Iscn/(exp(Vocn/(a.*vt))-1);  %Unadjusted model equation 1
%Io= Ion.*((Tn/T)^3).*exp(((q*1.12)/(a.*k)).*((1/Tn)-(1/T))); %Unadjusted 
%Model equation 2



for index_1= 1:length(V_pvt)
    V_ini = V_pvt(index_1);
    I= -10;                          %Temporary current to ensure while entry
    if(I_sol > 0)
        while abs(I_sol-I)>1e-8;
            I= Ipv - Io.*(exp((V_ini+Rs.*I_sol)/(vt.*a))-1)-(V_ini+Rs.*I_sol)/Rp; %Implicit I PV formula
            I_sol= I_sol + 1.*(I-I_sol); %Newton Raphson solution to PV current formula
        end
    end
    I_pvt(index_1) = I_sol;
end

%Outputs
V_pvt= V_pvt(I_pvt);
I_pvt= I_pvt(I_pvt);
P_pvt= I_pvt.*V_pvt;