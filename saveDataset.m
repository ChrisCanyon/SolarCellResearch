N = 2;

while N < 3
    tic
    disp("Generating Dataset for N = " + string(N));
    s = 100;%in 1000s
    
    [inputs, labels] = generateDataset(N,s*1000, 8);

    disp("Generation Finished. Saving Dataset");
    save("datasets/N" + string(N)  + "dataset" + string(s) + "k");
    N = N + 1;
    
    x = toc;
    disp("Time Elapsed: " + string(x/60) + " minutes")
    disp("  Time per N: " + string((x/60)/N) + " minutes")
    disp("  Time per 1k datapoints: " + string((x/60)/s) + " minutes")
    disp("  Time per N*size: " + string((x/60)/(s*N)) + " minutes")
end