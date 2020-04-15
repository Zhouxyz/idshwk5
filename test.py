from sklearn.ensemble import RandomForestClassifier
import math

domainlist = []
class Domain:
	def __init__(self,_domain,_label):
		self.domain = _domain
		self.label = _label


	def returnData(self):
		return [len(self.domain), getDigit(self.domain),getEntropy(self.domain)]

	def returnLabel(self):
		if self.label == "notdga":
			return "notdga"
		else:
			return "dga"
		
def initData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			domain = tokens[0]
			label = tokens[1]
			domainlist.append(Domain(domain,label))

	
def getDigit(string):
	count=0
	for i in string:
		if i.isdigit():
			count += 1
	return count
	
def getEntropy(string):
	str_list=list(string)
	n=len(str_list)
	str_list_single=list(set(str_list))
	num_list=[]
	for i in str_list_single:
   		num_list.append(str_list.count(i))

	list_two=zip(str_list_single,num_list)
	entropy=0
	for j in range(len(str_list_single)):
    		entropy+=-1*(float(num_list[j]/n))*math.log(float(num_list[j]/n),2)
	return entropy

def main():
	initData("train.txt")
	featureMatrix = []
	labelList = []
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())
	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	f=open("test.txt")
	fo=open("result.txt","w+")
	for line in f:
		line = line.strip()
		if line.startswith("#") or line =="":
			continue
		domain = line
		a=str(clf.predict([[len(domain),getDigit(domain),getEntropy(domain)]])).strip()	
		if a == "['notdga']":
			fo.write(domain+",notdga"+'\n')
		if a =="['dga']":
			fo.write(domain+",dga"+'\n')
	 	
if __name__ == '__main__':
	main()


