import csv
import numpy as np

# Read data into matrix
def loadData(filename):
	with open(filename,'rU') as csvfile:
		reader=csv.reader(csvfile)
		S = [[str(i) for i in r] for r in reader]
	return S

# Compare and merge new battle record and exclude
# those same battle records
def compare(S1, S2):
	newS = [];

	for line in S2:
		if not line in S1:
			newS.append(line);

	return newS;

def main(argv):
	file1 = str(argv[0]);
	if file1!='data_all.csv':
		return;
	S1 = loadData(file1);
	file2 = str(argv[1]);
	S2 = loadData(file2);
	print len(S2)
	newS = compare(S1, S2);
	print len(newS);
	file = open(file1, 'a');
	wr = csv.writer(file, dialect='excel');
	for line in newS:
		wr.writerow(line);

if __name__ == "__main__":
    main(sys.argv[1:]);