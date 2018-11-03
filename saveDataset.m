N = 5;
while N < 100
    size = 100;%in 1000s
    [inputs, labels] = generateDataset(N,size*1000);

    save("datasets/N" + string(N)  + "dataset" + string(size) + "k");
    N = N + 1;
end