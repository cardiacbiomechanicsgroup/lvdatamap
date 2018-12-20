function [L,M,T] = Cart2PSC(X,Y,Z,focus)
%
% Transformation from rectangular Cartesian to 
% prolate spheroid coordinates.  This is taken
% from https://cmrg.ucsd.edu:443/cvsweb/cont5/fe01/zx.f
% on the continuity website.
% z1=lambda, z2=mu, z3=theta
%
% NOTE: X-axis is base to apex

matrix_flag = false;
if size(X,2) > 1
    [rws,cls,pgs] = size(X);
    X = reshape(X,numel(X),1);
    Y = reshape(Y,numel(X),1);
    Z = reshape(Z,numel(X),1);
    matrix_flag = true;
end

L = zeros(length(X),1); M = L; T = L;
for jz = 1:length(X);
    x1 = X(jz);
    x2 = Y(jz);
    x3 = Z(jz);
    
        % Constants
        A1=(x1^2)+(x2^2)+(x3^2)-(focus^2);
        A2=sqrt((A1^2)+4*(focus^2)*((x2^2)+(x3^2)));
        A3=2*(focus^2);
        A4=max((A1+A2)/A3,0);
        A5=max((A2-A1)/A3,0);
        A6=sqrt(A4);
        A7=min(sqrt(A5),1);

        if abs(A7)<=1
            A8=asin(A7);
        else
            A8=0;
        end

        if x3==0 || A6==0 || A7==0
            A9=0;
        else
            if abs(A6*A7)>0
                A9=x3/(focus*A6*A7);
            else
                A9=0;
            end
        end

        if A9>=1
            A9=pi/2;
        elseif A9<=-1
            A9=-pi/2;
        else
            A9=asin(A9);
        end

        % Assign values
        z1=log(A6+sqrt(A4+1));

        if x1>=0
            z2=A8;
        else
            z2=pi-A8;
        end

        if x2>=0
            z3=mod((A9+2*pi),(2*pi));
        else
            z3=pi-A9;
        end
    
L(jz) = z1;
M(jz) = z2;
T(jz) = z3;

end

if matrix_flag
    L = reshape(L,rws,cls,pgs);
    M = reshape(M,rws,cls,pgs);
    T = reshape(T,rws,cls,pgs);
end
end