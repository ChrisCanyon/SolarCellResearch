N = 1;
i = 0;
while i < 100
    tic
    disp("Generating Dataset i = " + string(i));
    s = 10;%in 1000s
    
    [inputs, labels] = generateDataset(N,s*1000, 8);

    disp("Generation Finished. Saving Dataset");
    save("datasets/N" + string(N)  + "dataset" + string(s) + "k" + string(i));
    
    s = 100;
    [inputs, labels] = generateDataset(N,s*1000, 8);
    disp("Generation Finished. Saving Dataset");
    save("datasets/N" + string(N)  + "dataset" + string(s) + "k" + string(i));
    
    
    i = i + 1;
    
    x = toc;
    disp("Time Elapsed: " + string(x/60) + " minutes")
end