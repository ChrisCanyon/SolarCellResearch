clear;

[inputs, labels] = generateDataset(2,5);

save('dataset');

clear;

[inputs, labels] = generateDataset(2,1000);

save('dataset1k');

clear;

[inputs, labels] = generateDataset(2,10000);

save('dataset');
 