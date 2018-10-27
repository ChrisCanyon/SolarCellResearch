[singleCellData, singleCellLabels] = generateDataset(1, 1000);
[doubleCellData, doubleCellLabels] = generateDataset(2, 1000);

singleError = 0;
doubleError = 0;

for i = 1:1000
    singleGuess = ANN_V_MPP_Estimator(singleCellData(i,2),singleCellData(i));
    doubleGuess = ANN_V_MPP_Estimator(doubleCellData(i,2),doubleCellData(i));
    singleError = singleError + ((singleGuess - singleCellLabels(i))^2);
    doubleError = doubleError + ((doubleGuess - doubleCellLabels(i))^2);
end

MSEsingle = singleError/1000;
MSEdouble = doubleError/1000;

disp("MSEsingle:");
disp(MSEsingle);
disp("MSEdouble:");
disp(MSEdouble);