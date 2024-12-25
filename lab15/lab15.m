x = [-2 1;0 1; 2 -1; -2 -1;]; 
target = [0; 0; 1; 0]; 
w = [1 0.5]; 
b = [0.9]; 

max_e = 0.01; 
n = 0.6;
epoch = 1; 

linear_activation = @(z) z;

for e = 1 : epoch
    for i = 1 : size(x, 1)
        new_input = w * x(i, :)' + b;
        output = linear_activation(new_input);
        e = target(i) - output;
        if abs(e) < max_e
            continue;
        end
        w = w + n * e * x(i, :);
        b = b + n * e;
    end
end

% Итоговые веса и смещение
disp('Новые веса: ');
disp(w);
disp('Новое смещение: ');
disp(b);