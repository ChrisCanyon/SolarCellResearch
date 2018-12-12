import sys

if len(sys.argv) < 2:
    print("Error: please provide the templatefile")
    exit()

templateFilename = sys.argv[1]

for j in range(1, 9):
    outputFilename = templateFilename + "-" + str(j) + ".script"

    lines = [line.rstrip('\n') for line in open(templateFilename)]
    with open(outputFilename, 'w') as slurmFile:
        lines[1] = "#SBATCH --time=23:59:00"
        lines[10] = lines[10] + " " + str(j - 1) 
        #TODO fix file name so input number and input file number are the same

        for line in lines:
            slurmFile.write(line + "\n")
        
    