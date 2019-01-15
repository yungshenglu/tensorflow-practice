## Assignment 2

### Abstract
Implement **Kd-Tree** and use **K-NN classifier** to analyze a data set.

---

### Environment
python 3.5.2 or later version

### Execute
$ `./run.sh ../train.csv ../test.csv > output.txt`

this output the k nearest neighbors' index (sort from nearest to farthest) of first three testing data and accuracy of the testing data as a .txt file. (k = 1, 5, 10, 100)

### Dataset
`train.csv` is an ecoli data with 300 instances and 9 attributes without column 0 (name of ecoli). 

Column 10 is the class of ecoli. 
There are 8 classes: cp, im pp, imU, om, omL, inL, imS.  
Attibute information: 
  1.  mcg: McGeoch's method for signal sequence recognition.
  2.  gvh: von Heijne's method for signal sequence recognition.
  3.  lip: von Heijne's Signal Peptidase II consensus sequence score.   
      Binary attribute.
  5.  chg: Presence of charge on N-terminus of predicted lipoproteins.  
      Binary attribute.
  6.  aac: score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins.
  7. alm1: score of the ALOM membrane spanning region prediction program.
  8. alm2: score of ALOM program after excluding putative cleavable signal regions from the sequence.
  
You can create a file named 'test.csv' as the test case. Its format is the same as 'train.csv' and the instances can be derived from 'train.csv'.

