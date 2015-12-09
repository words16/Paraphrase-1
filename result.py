# -*- coding: utf-8 -*-


if __name__ == '__main__':
	ALL = 1.0
	TP = 0.0
	FP = 0.0

	FN = 0.0
	TN = 0.0

	t = 0.66
	for line in open("log2.txt").readlines():
		line = line.strip()
		line_split = line.split(" ")
		if len(line_split) == 2:
			ALL += 1
			flag = int(line_split[0])
			value = float(line_split[1].split("#")[1])
			if flag == 1:
			 # 	TP += 1
				if value >= t:
					TP += 1
				else:
					FP += 1
			else:
				if value >= t:
					FN += 1
				else:
					TN += 1


	print TP,FP,TN,FN
	A = (TP + TN) / (TP+FP+TN+FN)
	P = TP / (TP + FP)
	R =  TP / (TP + FN)
	F1 = 2*P*R / (P + R)

	print A,P,R,F1