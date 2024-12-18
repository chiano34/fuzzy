n = 6;

% Матрица парных сравнений для прибыли 
matrix_money = [1,   1/2,   3,    1/2,  1/3,   4,  
               2,    1,     4,    2,    1/3,   5, 
               1/3,  1/4,   1,    1/3,  1/5,   2,  
               2,    2,     3,    1,    1/3,   4, 
               3,    3,     5,    3,    1,     5, 
               1/4,  1/5,   1/2,  1/4,  1/5,   1];

% Матрица парных сравнений для себестоимости единицы продукции
matrix_sep = [ 1,    3,     1/2,  2,    4,     2,  
               1/3,  1,     1/4,  1/2,  2,     1/2, 
               2,    4,     1,    3,    4,     3,  
               1/2,  2,     1/3,  1,    3,     1, 
               1/4,  1/2,   1/4,  1/3,  1,     1/3, 
               1/2,  2,     1/3,  1,    3,     1];

% Матрица парных сравнений для доходов
matrix_income = [1,    2,     5,    3,    1,     4,  
                1/2,  1,     4,    2,    1/2,   3, 
                1/5,  1/4,   1,    1/3,  1/4,   1/2,  
                1/3,  2,     3,    1,    1/3,   2, 
                1,    3,     4,    3,    1,     3, 
                1/4,  1/5,   2,    2,    1/3,   1];

% Вычисление весов
weights_money = calculate_weights(matrix_money);
weights_sep = calculate_weights(matrix_sep);
weights_income = calculate_weights(matrix_income);
disp('Прибыль:');
disp(weights_money);
disp('Себестоимость продукции:');
disp(weights_sep);
disp('Доходы:');
disp(weights_income); 

%вычисление финального результата
final_scores = zeros(n, 1);
for i = 1 : n
    final_scores(i) = weights_money(i)+weights_sep(i)+weights_income(i);
end

disp("Оценка каждого предприятия:"+final_scores);
[max_score, best] = max(final_scores);
fprintf('Оптимальный вариант: Двигатель %d с весом: %.4f\n', best, max_score);

figure;
bar(final_scores);
xlabel('Варианты предприятий');
ylabel('Итоговый вес');
title('Итоговые веса вариантов предприятий');
xticklabels({'предприятие 1', 'предприятие 2', 'предприятие 3', 'предприятие 4', 'предприятие 5', 'предприятие 6'});
grid on;

% Функция для вычисления весов из матрицы парных сравнений
function [normalized_weights] = calculate_weights(matrix)
    n = size(matrix, 1);
    row_products = prod(matrix, 2);
    row_n_products = nthroot(row_products, n);
    total_sum = sum(row_n_products);
    disp(row_n_products)
    disp(total_sum)
    normalized_weights = row_n_products / total_sum;
end
%Лабораторная работа 7
products={'Кольцо','Ожерелье','Кулон','Перстень','Браслет','Цепь'}

% Матрица парных сравнений для золота(насколько хорошо выглядит чисто золотой
% вариант этого украшения от 1 до 20)=20,15,10,5,12,18
matrix_gold = [1,   1/3,   1/5,    1/7,  1/4,   1/2,  
               3,    1,     1/3,    1/5,  1/3,   2, 
               5,    3,     1,      1/3,  2,     4,  
               7,    5,     3,      1,    3,     5, 
               4,    3,     1/2,    1/3,  1,     3, 
               2,    2,     1/4,    1/5,  1/3,   1];
% Матрица парных сравнений для серебра(насколько хорошо выглядит чисто
% серебрянный
% вариант этого украшения от 1 до 20)=18,15,8,8,15,20
matrix_silver = [1,    1/2,    1/5,   1/5,  1/2,   2,  
                 2,    1,     1/4,    1/4,  1,     3, 
                 5,    4,     1,      1,    4,     6,  
                 5,    4,     1,      1,    4,     6, 
                 2,    1,     1/4,    1/4,  1,     3, 
                 1/2,  1/3,   1/6,    1/6,  1/3,   1];
% Матрица парных сравнений для роскоши(насколько дорого выглядит
% это украшение)=15,20,12,20,10,8
matrix_rich = [  1,    3,     1/2,  3,  1/3, 1/4,  
                 1/3,  1,     1/4,  1,  1/5, 1/5, 
                 2,    4,     1,    4,  1/2, 1/3,  
                 1/3,  1,   1/4,  1,  1/5, 1/5, 
                 3,    5,     2,    5,  1,   1/2, 
                 4,    5,     3,    5,  2,   1];
% Матрица парных сравнений для стоимости(насколько хорошо выглядит чисто
% серебрянный
% вариант этого украшения от 1 до 20)=18,15,8,8,15,20
matrix_cost = [1,    1/2,    1/5,   1/5,  1/2,   2,  
                 2,    1,     1/4,    1/4,  1,     3, 
                 5,    4,     1,      1,    4,     6,  
                 5,    4,     1,      1,    4,     6, 
                 2,    1,     1/4,    1/4,  1,     3, 
                 1/2,  1/3,   1/6,    1/6,  1/3,   1];
% Матрица парных сравнений для серебра(насколько дорого выглядит
% это украшение)=18,15,8,8,15,20
matrix_man = [1,    1/2,    1/5,   1/5,  1/2,   2,  
                 2,    1,     1/4,    1/4,  1,     3, 
                 5,    4,     1,      1,    4,     6,  
                 5,    4,     1,      1,    4,     6, 
                 2,    1,     1/4,    1/4,  1,     3, 
                 1/2,  1/3,   1/6,    1/6,  1/3,   1];
% Матрица парных сравнений для серебра(насколько хорошо выглядит чисто
% серебрянный
% вариант этого украшения от 1 до 20)=18,15,8,8,15,20
matrix_woman = [1,    1/2,    1/5,   1/5,  1/2,   2,  
                 2,    1,     1/4,    1/4,  1,     3, 
                 5,    4,     1,      1,    4,     6,  
                 5,    4,     1,      1,    4,     6, 
                 2,    1,     1/4,    1/4,  1,     3, 
                 1/2,  1/3,   1/6,    1/6,  1/3,   1];