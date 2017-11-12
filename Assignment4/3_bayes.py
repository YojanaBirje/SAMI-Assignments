import csv
import random
import math
import numpy as np
import math
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def igonre_col(data):
	columns=[]
	for i in range(0,len(data)):
		k=0
		for j in data[i]:
			if len(columns)-1<k:
				columns.append([])
			columns[k].append(j)
			k=k+1
	del(columns[24])
	length=len(columns[0])
	data=[]
	for i in range(0,length):
		l=[]
		for j in range(0,41):
			l.append(columns[j][i])
		data.append(l)
	return data
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	dataset1=[]
	for i in range(len(dataset)):
		for x in dataset[i]:
			dataset1.append(x)
	dataset=[]
	for i in range(0,len(dataset1),42):
		l=[]
		for j in range(i,i+42):
			l.append(dataset1[j])
		dataset.append(l)
	return dataset
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1].strip() not in separated):
			separated[vector[-1].strip()] = []
		separated[vector[-1].strip()].append(vector)
	return separated
def summarize(data):
	columns=[]
	for i in range(0,len(data)):
		k=0
		for j in data[i]:
			if len(columns)-1<k:
				columns.append([])
			columns[k].append(j)
			k=k+1
	
	length=len(columns[0])
	mean_mode=[]
	std_dev_list=[]
	col_dict=[]
	dict_nominal={}
	no_of_missing=0
	# del(columns[24])
	
	for i in range(0,39):
		sum=0.0
		dict_nominal={}
		if features1[i]=='continuous':
			for j in range(0,length):
				if columns[i][j].strip() !="?":
					no_of_missing=no_of_missing+1
					sum=sum+float(columns[i][j])
			mean=sum/length
			mean_mode.append(mean)
			std_dev=0.0
			for j in range(0,length):
				if columns[i][j].strip() !="?":
					no_of_missing=no_of_missing+1
					std_dev=std_dev+(float(columns[i][j])-mean)**2
			std_dev=(1.0*std_dev)/length
			std_dev=sqrt(std_dev)
			std_dev_list.append(std_dev)
			col_dict.append(dict_nominal)
		else:
			max_freq_val=""
			max_freq=0
			for j in range(0,length):
				if columns[i][j] not in dict_nominal and columns[i][j].strip()!='?':
					dict_nominal[columns[i][j]]=0
				if columns[i][j].strip()!='?':
					dict_nominal[columns[i][j]]=int(dict_nominal[columns[i][j]])+1
					if dict_nominal[columns[i][j]]>max_freq:
						max_freq=dict_nominal[columns[i][j]]
						max_freq_val=columns[i][j]
			mean_mode.append(max_freq_val)
			col_dict.append(dict_nominal)
			std_dev_list.append("--")
	# for i in range(0,39):
	# 	print "mean/mode = ",mean_mode[i],"		std_dev = ",std_dev_list[i]
	summary=[]
	summary.append(mean_mode)
	summary.append(col_dict)
	summary.append(std_dev_list)
	summary.append(length)
	summary.append(no_of_missing)
	return summary

def handle_missing(data,summary):
	columns=[]
	length_data=len(data)
	for i in range(0,len(data)):
		k=0
		for j in data[i]:
			if len(columns)-1<k:
				columns.append([])
			columns[k].append(j)
			k=k+1
	# del(columns[24])
	length=len(columns[0])
	for i in range(0,40):	
		for j in range(0,length):
			if columns[i][j].strip() == "?":
				columns[i][j]=summary[0][i]
	data=[]
	for i in range(0,length_data):
		l=[]
		for j in range(0,41):
			l.append(columns[j][i])
		data.append(l)
	# print data
	return data

def cal_prob(data,pr_c1,pr_c2):

	summary=[]
	seperated=separateByClass(data)
	for classValue, instances in seperated.iteritems():
		# print "class ",classValue
		summary1 = summarize(instances)
		summary.extend(summary1)
	count=0
	# ans_pr=[]
	for i in range(0,len(data)):
			pr1=0.0
			pr2=0.0
			for j in range(0,39):
					if data[i][j] in summary[1][j]:
						pr1=pr1+np.log(float(summary[1][j][data[i][j]])/float(summary[3]))
					if data[i][j] in summary[6][j]:
						pr2=pr2+np.log(float(summary[6][j][data[i][j]])/float(summary[8]))
			pr1=pr1+np.log(pr_c1)
			pr2=pr2+np.log(pr_c2)
			# print pr1,pr2
			if pr1>pr2 and data[i][-1].strip()=='50000+.':
				count=count+1
			if pr1<pr2 and data[i][-1].strip()=='- 50000.':
				count=count+1
	acc=100.0*count/len(data)
	# print "accuracy = ",acc
	return acc
def getclass(data):
	class_list=[]
	for i in data:
		class_list.append(i[-1])
	return class_list

features1=[]
def main():
	for i in range(0,42):
		if i in (0,5,16, 17,18,29,38):
			features1.append('continuous')
		else:
			features1.append('nominal')
	
	filename = 'census-income.data'
	dataset_read = loadCsv(filename)
	dataset_read = np.array(dataset_read)
	avg=0.0
	final_avg=0.0
	for k in range(0,30):
		kf = KFold(n_splits=10,shuffle=True)
		kf.get_n_splits(dataset_read)
    	
		acc=0.0
		for train_index, test_index in kf.split(dataset_read):
			dataset, dataset_test = dataset_read[train_index], dataset_read[test_index]

			seperated = separateByClass(dataset)

			# filename = 'census-income.test'
			# dataset_test = loadCsv(filename)
			seperated_test = separateByClass(dataset_test)
			l1=len(seperated_test['50000+.'])
			l2=len(seperated_test['- 50000.'])
			# summarize

			for classValue, instances in seperated.iteritems():
				instances=igonre_col(instances)
				summary = summarize(instances)
				# print "class ",classValue," total examples = ",len(instances)," no. of missing entries = ",summary[4]
				# for i in range(0,39):
				# 	print "mean/mode = ",summary[0][i],"		std_dev = ",summary[2][i]

			
			data=[]
			for classValue, instances in seperated_test.iteritems():
				instances=igonre_col(instances)
				# print "class ",classValue
				data1=[]
				data1 = handle_missing(instances,summary)
				data.extend(data1)
			pr_c1=1.0*l1/len(dataset_test)
			pr_c2=1.0*l2/len(dataset_test)
			acc=acc+cal_prob(data,pr_c1,pr_c2)
		avg=avg+acc/10
		print "accuacy = ",1.0*acc/10
	final_avg=avg/30
	print "Cross Validaton Accuracy = ",final_avg
main()
