function [t,y]=ChaosLibrary(S,t,IC,p)
% function ChaosLibrary20160700(S,t,IC,p)
% inputs  - S, string name of chaotic attractor
%         - t, time, either of the form [to,tf] or [to:f:tf]
%         - IC, initial conditions
%         - p, coefficients of the system of differential equations used.
% outputs - t, time
%         - y, column oriented time series
% Remarks
% - This MATLAB m file uses ODE solvers to calculate various chaotic 
%   attractors. The user is able to change all components of the chaotic 
%   attractor, including the time scale, the initial conditions and the 
%   internal parameters of the attractor equations. The m file will
%   also sketch out the plot bfore displaying a final version of it so the
%   user will be able to see the path a particle would take, and observe
%   patterns.
% - The first function handle contains the code that governs the input
%   recognition from the user. The rest of the handles contain more
%   information on the chaotic attractors, and includes the vector
%   components and typical parameters and initial conditions.
% - For 2D attractors, the t variable is used to specify the number of data
%   points. The rest of the function inputs can be used normally. 
% Future Work
% - More systems could be added.
% Jun 2016 - Created by Christopher Cunningham
% Jul 2016 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Reformated comments section.
%          - Hennon, Logistic, Aizawa attractors added.
% Jul 2021 - Modified by Ben Senderling, bmchnonan@unomaha.edu
%          - Removed plotting.
% Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
% Variability, University of Nebraska at Omaha
%
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright 
%    notice, this list of conditions and the following disclaimer in the 
%    documentation and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its 
%    contributors may be used to endorse or promote products derived from 
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%% Begin code

switch S
    case 'Rossler'
        [t,y] = ode45(@Rossler,t,IC,[],p);
    case 'Lorenz'
        [t,y] = ode45(@Lorenz,t,IC,[],p);
    case 'Hennon'
        x(1) = IC(1);
        y(1) = IC(2);
        for c = 1:t
            x(c+1) = 1 - p(1)*x(c)^2 + y(c);
            y(c+1) = p(2)*x(c);
        end
    case 'Logistic'
        x(1) = IC(1);
        for c = 1:t
            x(c+1) = p(1)*x(c)*(1-x(c));
        end
    case 'Aizawa'
        [t,y] = ode23(@Aizawa,t,IC,[],p);
    case 'DequanLi'
        
        [t,y] = ode23(@DequanLi,t,IC,[],p);
    case 'NoseHoover'
        [t,y] = ode45(@NoseHoover,t,IC,[],p);
    case 'QiChen'
        [t,y] = ode45(@QiChen,t,IC,[],p);
    case'YuWang'
        [t,y] = ode45(@YuWang,t,IC,[],p);
    case 'LuChen'
        [t,y] = ode45(@LuChen,t,IC,[],p);
    case 'Arneodo'
        [t,y] = ode45(@Arneodo,t,IC,[],p);
    case 'TSUCSI'
        [t,y] = ode45(@TSUCSI,t,IC,[],p);
end
end

function [yp] = Rossler(t,y,p)
% function [yp] = Rossler(t,y,p)
% inputs  - 
% outputs - 
% Remarks
% - Typical IC's are: [-9 0 0]
% - Typical Parameters are: [0.2 0.2 5.7]
% - Source: https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
% - Equations
%       x'= -y -z
%       y'= x + ay
%       z'= b+ z(x-c)
%% Begin code

yp = zeros(3,1);
yp(1) = -y(2)-y(3);
yp(2) = y(1) + p(1)*y(2);
yp(3) = p(2) + y(3)*(y(1)-p(3));

end

%LORENZ================================================
function [ yp ] = Lorenz(t,y,p)
%   Typical IC's are: 0 -0.01 9
%   Typical Params are: 10 28 8/3
%   Source: https://en.wikipedia.org/wiki/Lorenz_system
%   EQUATION
%       x' = o(y-x)
%       y' = x(p-z)-y
%       z' = xy - Bz
%% Begin code

yp = zeros(3,1);
yp(1) = p(1)*( y(2) - y(1) );
yp(2) = -y(1).*y(3) + p(2).*y(1) - y(2);
yp(3) = (y(1).*y(2)) - p(3)*y(3);

end

%DEQUAN-LI=============================================
function [ yp ] = DequanLi(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 40 0.16 55 20 1.833 0.65
%   Source: https://www.researchgate.net/publication/223710675_A_three-scroll_chaotic_attractor
%   EQUATION
%       x' =  a(y-x)+dxz
%       y' = px + Ly - xz
%       z = Bz + xy = ex^3
%% Begin code

yp = zeros(3,1);
yp(1) = p(1)*(y(2) - y(1)) + p(2)*(y(1)*y(3));
yp(2) = p(3)*y(1) + p(4)*y(2) - y(1)*y(3);
yp(3) = p(5)*y(3) + y(1)*y(2) - p(6)*(y(1)*y(1));
end

%NOSE-HOOVER===========================================
function [ yp ] = NoseHoover(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 1.5
%   Source: http://williamhoover.info/Scans1980s/1986-4.pdf
%   EQUATION:
%       x' = y
%       y' = -x +yz
%       z' = a - y^2
%% Begin code

yp = zeros(3,1);
yp(1) = y(2);
yp(2) = -y(1) + y(2)*y(3);
yp(3) = p(1) - (y(2)*y(2));
end

%QI-CHEN===============================================
function [ yp ] = QiChen(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 38 8/3 80
%   Source: https://www.emis.de/journals/HOA/MPE/Volume2012/438328.pdf
%   EQUATION
%       x' = a(y-x) + yz
%       y' = cx + y -xz
%       z' = xy - Bz
%% Begin code

yp = zeros(3,1);
yp(1) = p(1)*(y(2)-y(1)) + y(2)*y(3);
yp(2) = p(3)*y(1) + y(2) - y(1)*y(3);
yp(3) = y(1)*y(2) - p(2)*y(3);
end

%YU-WANG===============================================
function [ yp ] = YuWang(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 10 40 2 2.5
%   EQUATION:
%       x'= a(y-x)
%       y' = Bx - cxz
%       z' = e^(xy) - dz
%% Begin code

yp = zeros(3,1);
yp(1) = p(1)*(y(2)-y(1));
yp(2) = p(2)*y(1) - p(3)*y(1)*y(3);
yp(3) = exp(y(1)*y(2)) - p(4)*y(3);
end

%LU-CHEN===============================================
function [ yp ] = LuChen(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: -10 -4 18.1
%   Source: https://en.wikipedia.org/wiki/Multiscroll_attractor
%   EQUATION
%       x' = -((aBx)/(a+B))-yz + c
%       y' = ay + xz
%       z' = Bz + xy
%% Begin code

yp = zeros(3,1);
yp(1) = -1*(p(1)*p(2)*y(1)/(p(1)+p(2))) - y(2)*y(3) + p(3);
yp(2) = p(1)*y(2) + y(1)*y(3);
yp(3) = p(2)*y(3) + y(1)*y(2);
end

%ARNEODO===============================================
function [ yp ] = Arneodo(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: -5.5 3.5 -1
%   EQUATION
%       x' = y
%       y' = z
%       z' = =ax -By - z + dx^3
%% Begin code

yp = zeros(3,1);
yp(1)= y(2);
yp(2) = y(3);
yp(3) = -p(1)*y(1) - p(2)*y(2)-y(3) + p(3)*y(1)*y(1)*y(1);
end

%TSUCSI================================================
function [yp] = TSUCSI(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 40 0.833 0.5 0.65 20
%   Source: https://www.researchgate.net/publication/265811017_A_New_Three-Scroll_Unified_Chaotic_System_Coined
%   EQUATION
%       x' = a(y-x) +dxz
%       y' = Ly - xz
%       z' = Bz + xy - ex^2
%% Begin code

yp = zeros(3,1);
yp(1) = p(1)*(y(2)-y(1)) + p(3)*y(1)*y(3);
yp(2) = p(5)*y(2) - y(1)*y(3);
yp(3) = p(2)*y(3) + y(1)*y(2) - p(4)*y(1)*y(1);
end

%AIZAWA================================================
function [yp] =Aizawa(t,y,p)
%   Typical IC's are: Unknown, start with a point other than [0 0 0]
%   Typical Params are: 0.95 0.9 0.6 3.5 0.25 0.1 
%   EQUATION
%       x' = (z-b)*x - d*y
%       y' = d*y + (z-b)*y 
%       z' = c + a*z - (z^3/3)+f*z*x^3
%% Begin code

yp=zeros(3,1);
yp(1)=(y(3)-p(2))*y(1)-p(4)*y(2);
yp(2)=p(4)*y(1)+(y(3)-p(2))*y(2);
yp(3)=p(3)+p(1)*y(3)-(y(3))^3/3-(y(1))^2+p(6)*y(3)*(y(1))^3;

end

%HENNON================================================
%   Typical IC's are: Unknown, start with a point other than [0 0]
%   Typical Params are: 1.4 0.3
%   Source: https://en.wikipedia.org/wiki/H%C3%A9non_map
%   EQUATION
%       x(n+1) = 1 - ax(n)^2 + y(n)
%       y(n+1) = bx(n)

%LOGISTIC==============================================
%   Typical IC is: Use a value between 0 and 1
%   Typical Params are: Any positive value, usually between 0 and 4
%   Source: https://en.wikipedia.org/wiki/Logistic_map
%   EQUATION
%       x(n+1) = r*x(n)*(1-x(n))




