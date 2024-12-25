training = [104.2361 106.1878 107.1758 107.7409 109.5782
        108.0104 105.0604 103.7908 102.5761 100.6798
        100.2192 100.0348 99.9430 99.9971 99.0180
        98.3657 97.9559 97.9550 97.8335 98.0726
        98.2236 98.0562 97.5499 97.4402 97.0226
        97.0530 97.3261 97.2300 96.6657 96.7402
        96.6379 96.5918 96.0924 96.4172 97.1490
        97.2568 97.0121 96.1021 96.0686 97.2394
        96.9483 96.1079 96.0649 94.8700 95.0262
        94.5054 93.3581 93.2221 103.3837 99.4215];
 disp(training);

min_vals = min(training);
max_vals = max(training);

input = training(:, 1:4);
 disp(input);

output = training(:, 5);
 disp(output);

prognoz = newff(minmax(input'), [15 1], { 'purelin' 'purelin'});
 disp(prognoz);

prognoz = train(prognoz, input', output');

test = [91.4548 91.4449 91.4807 91.7745,91.6012
        91.2881 91.6862 90.6944 89.5428 88.9062
        88.7960 90.0055 92.6592 89.9475 87.9920
        86.5621 85.9543 85.1646 84.9471 85.7024];

test_input = test(:, 1:4);
 disp("test input");
 disp(test_input);

test_output = test(:, 5);
disp("test_output");
disp(test_output);

pred = sim(prognoz, test_output');
disp('prediction')
disp(pred);
 fis=readfis('Anfis.fis')
predict=evalfis(fis,[96.9483 96.1079 96.0649 94.8700])
disp(predict)