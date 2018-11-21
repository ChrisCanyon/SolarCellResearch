import sys
from parse import parse
import csv

if len(sys.argv) < 2:
    print("Error: specify file")
    exit()

filename = sys.argv[1]

lines = [line.rstrip('\n') for line in open(filename)]

with open(filename + '.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["CellsPerSensor", "MSE", "Weight&Biases", "ObjectiveFunctionValue", "Architecture"]) 
    for i in range(0, len(lines), 8):
        CellsPerSensor = lines[i+1]
        Architecture = lines[i+2]
        MSE = lines[i+3]
        WeightsAndBiases = lines[i+4]
        ObjectiveFunctionValue = lines[i+5]

        C = parse("CellsPerSensor: {}" , CellsPerSensor)
        A = parse(" Resulting Network Architecture: {}" , Architecture)
        M = parse(" MSE after 5-Fold CV: {}" , MSE)
        W = parse(" Weights and Biases: {}" , WeightsAndBiases)
        O = parse(" Objective Function Value: {}" , ObjectiveFunctionValue)
        writer.writerow([C[0], M[0], W[0], O[0], A[0]])

 