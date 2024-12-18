%x = 0:0.01:7;
%y(x<=5)=1
%y(x > 5) =1-( 1 ./ (1 + exp(-10 * (x(x > 5) - 6))));
%plot(x,y);
x = 0 : 0.1 : 10;
mf1 = trimf(x, [7 9.5 12]);
mf2 = gaussmf(x, [2 6]);
x1 = (0 : 0.1 : 10);
x2 = (2 : 0.1 : 10);
[X, Y] = meshgrid(x1, x2);
Z = min(trimf(X, [7 9.5 12]), gaussmf(Y, [2 6]));

plot3(X,Y,Z)