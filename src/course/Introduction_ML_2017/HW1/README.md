## Asssignment 1

For	this	assignment, you	need	to	implement **ID3	algorithm** to	construct	a	
decision	tree with	C,	C++,	Java	or	python2/3,	and	use	**K-fold	cross	validation**	
(K=5)	to	validate	classification	performance	by	outputting precision	and	recall	
for	each	class and	total	accuracy.

---
### Environment
  - python 3.5.2 or later version
  
### Usage
`./run.sh` - executes `ID3.py` and shows the output of accuracy, precision and recall.

`./RF.sh` - executes `RF.py` and shows the output of accuracy, precision and recall but using *Random Forest* mechanism.

### Dataset
https://archive.ics.uci.edu/ml/datasets/Iris

Including	150	number	of	instances	with	4	attributes.

### Output Format
The	accuracy,	precision	and	recall	are	
floating	numbers	within	0 and	1 and	arranged	with	the	following	format
```
[Total accuracy]	
[Precision of class 0] [Recall of class 0]	
[Precision of class 1] [Recall of class 1]	
[Precision of class 2] [Recall of class 2]
```
