N = 12;

while N < 1000
    size = 100;%in 1000s
    [inputs, labels] = generateDataset(N,size*1000);

    save("datasets/N" + string(N)  + "dataset" + string(size) + "k");
    N = N + N*25;
end