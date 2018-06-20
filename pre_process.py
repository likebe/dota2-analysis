### display.py
### project 9
### Kebing Li
### 05/10/2018
### CS251

import csv
import random

def pre_process(file):
	file2 = file[:-4]+'_data.csv'
	myFile = open(file2, 'w')
	with myFile:
		writer = csv.writer(myFile)
		list = ['Won', 'Cluster ID', 'Game Mode', 'Game Type']
		for i in range(113):
			string = 'Hero ID: ' + str(i+1)
			list.append(string)
		writer.writerow(list)
		list2 = []
		for i in range(117):
			list2.append("numeric")
		writer.writerow(list2)

		# with open(file, newline='') as csvfile:
		# 	reader = csv.reader(csvfile)
		# 	total_row = sum(1 for row in reader)
		# 	half_row = int(total_row/2)
		# 	i = 0
		# 	for row in reader:
		# 		if i < half_row:
		# 			print(row)
		# 			print('yes')
		# 			writer.writerow(row)
		# 		else:
		# 			pass
		# 		i = i + 1
		with open(file, newline='') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if random.random() > 0.5:
					writer.writerow(row)


def test():
	pre_process('dota2Train.csv')
	pre_process('dota2Test.csv')

if __name__ == "__main__":
	test()